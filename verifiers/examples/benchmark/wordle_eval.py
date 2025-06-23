"""Wordle evaluation benchmark using willcb/V3-wordle dataset."""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import random

from verifiers.envs import MultiTurnEnv
from verifiers.rubrics import Rubric
from verifiers.parsers import Parser


class WordleEvalEnv(MultiTurnEnv):
    """Wordle evaluation environment for benchmarking with HuggingFace dataset."""
    
    def __init__(self, target_word: str = None, word_list: List[str] = None):
        super().__init__()
        self.target_word = target_word
        self.word_list = word_list or []
        self.max_attempts = 6
        self.current_attempt = 0
        self.guesses = []
        self.feedback_history = []
    
    async def reset(self, target_word: Optional[str] = None) -> str:
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
    
    async def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Process a guess and return feedback."""
        guess = action.upper().strip()
        
        if len(guess) != 5 or not guess.isalpha():
            return "Please enter a valid 5-letter word.", 0.0, False, {}
        
        self.current_attempt += 1
        self.guesses.append(guess)
        
        if guess == self.target_word:
            reward = 1.0 - (self.current_attempt - 1) / self.max_attempts
            return (
                f"<� Excellent! You found '{self.target_word}' in {self.current_attempt} attempt{'s' if self.current_attempt > 1 else ''}!\n"
                f"Final score: {reward:.2f}"
            ), reward, True, {"success": True, "attempts": self.current_attempt}
        
        feedback = self._get_feedback(guess)
        self.feedback_history.append(feedback)
        done = self.current_attempt >= self.max_attempts
        
        if done:
            return (
                f"{feedback}\n\n"
                f"Game over! The word was '{self.target_word}'.\n"
                f"Better luck next time!"
            ), 0.0, True, {"success": False, "attempts": self.current_attempt}
        
        remaining = self.max_attempts - self.current_attempt
        history = "\n".join([f"Guess {i+1}: {g} � {f}" 
                           for i, (g, f) in enumerate(zip(self.guesses, self.feedback_history))])
        
        return (
            f"Current guess: {guess}\n"
            f"Feedback: {feedback}\n\n"
            f"History:\n{history}\n\n"
            f"You have {remaining} attempt{'s' if remaining > 1 else ''} remaining.\n"
            f"Please make your next guess."
        ), 0.0, False, {}
    
    def _get_feedback(self, guess: str) -> str:
        """Generate color-coded feedback for the guess."""
        feedback = []
        target_chars = list(self.target_word)
        guess_chars = list(guess)
        
        # First pass: mark correct positions
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback.append(f"=�{guess_chars[i]}")
                target_chars[i] = None
                guess_chars[i] = None
        
        # Second pass: mark correct letters in wrong positions
        for i in range(5):
            if guess_chars[i] is None:
                continue
            elif guess_chars[i] in target_chars:
                feedback.insert(i, f"=�{guess_chars[i]}")
                target_chars[target_chars.index(guess_chars[i])] = None
            else:
                feedback.insert(i, f"{guess_chars[i]}")
        
        return "".join(feedback)


class WordleEvalRubric(Rubric):
    """Rubric for evaluating Wordle performance."""
    
    async def score(self, trajectory: List[Dict[str, Any]]) -> float:
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
        # Look for patterns like "guess: WORD" or "my guess is WORD"
        import re
        
        # Try to find explicit guess patterns
        patterns = [
            r"guess[:\s]+([A-Za-z]{5})",
            r"my guess is[:\s]+([A-Za-z]{5})",
            r"I'll guess[:\s]+([A-Za-z]{5})",
            r"trying[:\s]+([A-Za-z]{5})",
            r"^([A-Za-z]{5})$",  # Just the word alone
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # Fallback: find any 5-letter word
        words = re.findall(r'\b[A-Za-z]{5}\b', response)
        if words:
            return words[0].upper()
        
        # Last resort: take first 5 alphabetic characters
        letters = ''.join(c for c in response if c.isalpha())
        return letters[:5].upper() if len(letters) >= 5 else "GUESS"


async def evaluate_wordle_dataset(
    model_name: str = "Qwen/Qwen3-1.7B",
    num_episodes: int = 100,
    dataset_name: str = "willcb/V3-wordle",
    split: str = "test"
):
    """Run Wordle evaluation on HuggingFace dataset."""
    from vllm import LLM, SamplingParams
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    # Extract target words from the dataset
    target_words = []
    for item in dataset:
        if "word" in item:
            target_words.append(item["word"].upper())
        elif "target" in item:
            target_words.append(item["target"].upper())
        elif "answer" in item:
            target_words.append(item["answer"].upper())
    
    if not target_words:
        print("Warning: No target words found in dataset. Using default word list.")
        target_words = ["HELLO", "WORLD", "PYTHON", "CODING", "NEURAL"]
    
    print(f"Found {len(target_words)} words in dataset")
    
    # Initialize vLLM model
    print(f"Initializing model: {model_name}")
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
        top_p=0.95
    )
    
    parser = WordleParser()
    rubric = WordleEvalRubric()
    
    results = []
    num_episodes = min(num_episodes, len(target_words))
    
    for i in range(num_episodes):
        target_word = target_words[i % len(target_words)]
        env = WordleEvalEnv(target_word=target_word)
        trajectory = []
        
        observation = await env.reset()
        done = False
        
        messages = [{"role": "user", "content": observation}]
        
        while not done:
            # Format messages for the model
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
            
            # Generate response
            outputs = llm.generate([prompt], sampling_params)
            model_response = outputs[0].outputs[0].text.strip()
            
            action = parser.parse(model_response)
            
            observation, reward, done, info = await env.step(action)
            
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
        
        score = await rubric.score(trajectory)
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
            print(f"Episodes {i-8}-{i+1}: Avg Score: {avg_score:.3f}, Success Rate: {success_rate:.3f}")
    
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


async def main():
    """Main evaluation function."""
    model = "Qwen/Qwen3-1.7B"
    dataset = "willcb/V3-wordle"
    num_episodes = 50
    
    print(f"Starting Wordle evaluation")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)
    
    results = await evaluate_wordle_dataset(
        model_name=model,
        dataset_name=dataset,
        num_episodes=num_episodes
    )
    
    analyze_results(results)
    
    # Save results
    with open("wordle_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to wordle_eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())