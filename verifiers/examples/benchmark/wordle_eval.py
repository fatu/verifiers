"""Wordle evaluation benchmark using willcb/V3-wordle dataset."""
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import random
from tqdm import tqdm

from vllm import LLM, SamplingParams

from verifiers.rubrics import Rubric
from verifiers.parsers import Parser


class WordleEvalEnv:
    """Wordle evaluation environment for benchmarking with HuggingFace dataset."""
    
    def __init__(self, target_word: str = None, word_list: List[str] = None):
        self.target_word = target_word
        self.word_list = word_list or []
        self.max_attempts = 6
        self.current_attempt = 0
        self.guesses = []
        self.feedback_history = []
    
    def reset(self, target_word: Optional[str] = None) -> str:
        """Reset the environment and return initial prompt."""
        self.current_attempt = 0
        self.guesses = []
        self.feedback_history = []
        
        if target_word:
            self.target_word = target_word.upper()
        elif not self.target_word and self.word_list:
            self.target_word = random.choice(self.word_list).upper()
            
        return (
            f"Let's play Wordle! I'm thinking of a 5-letter word.\n"
            f"You have {self.max_attempts} attempts to guess it.\n"
            f"After each guess, I'll give you feedback:\n"
            f"=� = correct letter in correct position\n"
            f"=� = correct letter in wrong position\n"
            f" = letter not in the word\n\n"
            f"Please make your first guess."
        )
    
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Process a guess and return feedback."""
        guess = action.upper().strip()
        
        self.current_attempt += 1
        self.guesses.append(guess)
        
        if guess == self.target_word:
            reward = 1.0 - (self.current_attempt - 1) / self.max_attempts
            return (
                f"<Excellent! You found '{self.target_word}' in {self.current_attempt} attempt{'s' if self.current_attempt > 1 else ''}!\n"
                f"Final score: {reward:.2f}"
            ), reward, True, {"success": True, "attempts": self.current_attempt}
        if len(guess) != 5 or not guess.isalpha():
            feedback = "Please enter a valid 5-letter word."
        elif guess == 'INVALID':
            feedback = "Invalid guess format. Please provide a 5-letter word inside <guess> tags."
        else:
            feedback = self._get_feedback(guess)
        self.feedback_history.append(feedback)
        done = self.current_attempt >= self.max_attempts
        
        if done:
            return (
                f"{feedback}\n\n"
                f"Game over! The word was '{self.target_word}'.\n"
            ), 0.0, True, {"success": False, "attempts": self.current_attempt}
        
        remaining = self.max_attempts - self.current_attempt
        
        # Format guess with spaces between letters to match feedback
        spaced_guess = ' '.join(guess)
        
        return (
            f"\n{spaced_guess}\n"
            f"{feedback}\n\n"
            f"You have {remaining} {'guesses' if remaining > 1 else 'guess'} left."
        ), 0.0, False, {}
    
    def _get_feedback(self, guess: str) -> str:
        """Generate feedback with G/Y/X indicators."""
        feedback = []
        target_chars = list(self.target_word)
        guess_chars = list(guess)
        
        # First pass: mark correct positions
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback.append('G')
                target_chars[i] = None
            else:
                feedback.append(None)
        
        # Second pass: mark correct letters in wrong positions
        for i in range(5):
            if feedback[i] == 'G':
                continue
            elif guess_chars[i] in target_chars:
                feedback[i] = 'Y'
                target_chars[target_chars.index(guess_chars[i])] = None
            else:
                feedback[i] = 'X'
        
        return ' '.join(feedback)


class WordleEvalRubric(Rubric):
    """Rubric for evaluating Wordle performance."""
    
    def score(self, trajectory: List[Dict[str, Any]]) -> float:
        """Score based on success and number of attempts."""
        if not trajectory:
            return 0.0
        
        final_info = trajectory[-1].get("info", {})
        if not final_info.get("success", False):
            return 0.0
        
        attempts = final_info.get("attempts", 6)
        # Score: 1.0 for 1 attempt, 0.8 for 2, ..., 0.2 for 6
        return max(0.2, 1.0 - (attempts - 1) * 0.16)


class WordleParser(Parser):
    """Parser for extracting Wordle guesses from model responses."""
    
    def parse(self, response: str) -> str:
        """Extract the 5-letter guess from the response."""
        import re
        
        # First check for <guess>...</guess> tags (may contain brackets)
        guess_match = re.search(r'<guess>\s*\[?([A-Za-z]{5})\]?\s*</guess>', response, re.IGNORECASE)
        if guess_match:
            return guess_match.group(1).upper()
        
        # If that fails, return whatever is in <guess> tags
        loose_guess_match = re.search(r'<guess>\s*\[?(.*?)\]?\s*</guess>', response, re.IGNORECASE | re.DOTALL)
        if loose_guess_match:
            return loose_guess_match.group(1).strip()

        return "INVALID" 

def evaluate_wordle_dataset(
    llm,
    sampling_params,
    num_episodes: int = 100,
    dataset_name: str = "willcb/V3-wordle",
    split: str = "test"
):
    """Run Wordle evaluation on HuggingFace dataset."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, download_mode="reuse_cache_if_exists")
    
    # Get tokenizer from the LLM
    tokenizer = llm.get_tokenizer()
    
    # Extract target words and prompts from the dataset
    target_words = []
    prompts = []
    for item in dataset:
        if "answer" in item:
            target_words.append(item["answer"].upper())
        
        # Get the prompt from the dataset
        if "prompt" in item:
            prompts.append(item["prompt"])
        else:
            prompts.append(None)
    
    if not target_words:
        raise ValueError(f"No target words found in dataset {dataset_name}. "
                        "Expected 'answer' field in dataset items.")
    
    print(f"Found {len(target_words)} words in dataset")
    
    parser = WordleParser()
    rubric = WordleEvalRubric()
    
    results = []
    num_episodes = min(num_episodes, len(target_words))
    
    for i in tqdm(range(num_episodes), desc="Evaluating episodes", unit="episode"):
        idx = i % len(target_words)
        target_word = target_words[idx]
        dataset_prompt = prompts[idx]
        
        env = WordleEvalEnv(target_word=target_word)
        trajectory = []
        
        # Use prompt from dataset if available
        env.reset()  # Reset the environment state
        
        if dataset_prompt and isinstance(dataset_prompt, list):
            # Use the formatted messages from dataset
            messages = dataset_prompt.copy()
        
        done = False
        
        while not done:
            # Apply chat template to format messages
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True
            )

            # Generate response
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            think_text = outputs[0].outputs[0].text.strip()
            # If </think> isn't present, we inject an early-stop signal

            if "<think>" in think_text and "</think>" not in think_text:
                early = "\n\nConsidering limited time, I must answer now.</think>\n\n<guess>"
                prompt2 = prompt + think_text + early
                sampling_params2 = sampling_params
                sampling_params2.max_new_tokens = 100  # e.g., 512
                outputs2 = llm.generate([prompt2], sampling_params2, use_tqdm=False)
                final_text = think_text + early + outputs2[0].outputs[0].text.strip()
            else:
                final_text = think_text
            
            model_response = final_text
            
            action = parser.parse(model_response)
            
            observation, reward, done, info = env.step(action)
            
            trajectory.append({
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "response": model_response
            })
            
            # Update conversation history
            messages.append({"role": "assistant", "content": model_response})
            messages.append({"role": "user", "content": observation})
        
        score = rubric.score(trajectory)
        results.append({
            "episode": i,
            "target_word": target_word,
            "score": score,
            "attempts": trajectory[-1]["info"].get("attempts", 6),
            "success": trajectory[-1]["info"].get("success", False),
            "guesses": [t["action"] for t in trajectory]
        })
        
        if (i + 1) % 10 == 0:
            recent_results = results[-10:]
            avg_score = sum(r["score"] for r in recent_results) / len(recent_results)
            success_rate = sum(r["success"] for r in recent_results) / len(recent_results)
            tqdm.write(f"Episodes {i-8}-{i+1}: Avg Score: {avg_score:.3f}, Success Rate: {success_rate:.3f}")
    
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and print detailed results."""
    total = len(results)
    successful = [r for r in results if r["success"]]
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"\nTotal episodes: {total}")
    print(f"Successful games: {len(successful)} ({len(successful)/total*100:.1f}%)")
    
    if successful:
        avg_attempts = sum(r["attempts"] for r in successful) / len(successful)
        print(f"Average attempts when successful: {avg_attempts:.2f}")
        
        # Distribution of attempts
        attempt_dist = {}
        for r in successful:
            attempts = r["attempts"]
            attempt_dist[attempts] = attempt_dist.get(attempts, 0) + 1
        
        print("\nAttempt distribution (successful games):")
        for attempts in sorted(attempt_dist.keys()):
            count = attempt_dist[attempts]
            percentage = count / len(successful) * 100
            print(f"  {attempts} attempts: {count} ({percentage:.1f}%)")
    
    avg_score = sum(r["score"] for r in results) / total
    print(f"\nOverall average score: {avg_score:.3f}")
    
    # Show some example games
    print("\nExample games:")
    for i, result in enumerate(results[:3]):
        print(f"\n  Game {i+1}: {result['target_word']}")
        print(f"  Success: {result['success']}")
        print(f"  Guesses: {' � '.join(result['guesses'])}")


def main():
    """Main evaluation function."""
    
    
    model = "outputs/model/sft-wordle/checkpoint-1500"
    # model = "Qwen/Qwen3-4B"
    dataset = "willcb/V3-wordle-test"
    num_episodes = 10
    thinking_budget = 512
    
    print(f"Starting Wordle evaluation")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    
    print(f"Initializing model: {model}")
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        # gpu_memory_utilization=0.9,
        max_model_len=24576,
        enforce_eager=True,
        )
    sampling_params = SamplingParams(
        max_tokens=thinking_budget,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )
    
    # Run async evaluation
    results = evaluate_wordle_dataset(
        llm=llm,
        sampling_params=sampling_params,
        dataset_name=dataset,
        num_episodes=num_episodes,
        split="train"
    )
    
    analyze_results(results)
    
    # Save results
    with open("wordle_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to wordle_eval_results.json")


if __name__ == "__main__":
    main()