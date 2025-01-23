import concurrent.futures
import time
from typing import List, Dict, Any
# import pytest

from test_client import synthesize_speech
from fish_speech.utils.schema import ServeTTSRequest

def generate_test_cases() -> List[Dict[str, Any]]:
    texts = [
        "Hello, this is a test case.",
        "Testing different voice parameters.",
        "Let's try another variation.",
        "This is a longer sentence to test synthesis capabilities.",
        "Short test."
    ]
    
    test_cases = []
    for i in range(50):
        test_case = {
            "text": texts[i % len(texts)],
            "normalize": i % 2 == 0,
            "format": "wav",
            "temperature": round(0.5 + (i % 6) * 0.1, 1),  # varies from 0.5 to 1.0
            "top_p": round(0.5 + (i % 5) * 0.1, 1),  # varies from 0.5 to 0.9
            "repetition_penalty": round(1.0 + (i % 4) * 0.1, 1),  # varies from 1.0 to 1.3
            "chunk_length": 150 + (i % 3) * 50,  # varies between 150, 200, 250
            "seed": i if i % 3 == 0 else None  # alternates between None and values
        }
        test_cases.append(test_case)
    
    print(f"Generated {len(test_cases)} test cases with varying parameters")
    return test_cases

def test_concurrent_synthesis():
    test_cases = generate_test_cases()[:50]  # Use fewer cases for synthesis test
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                synthesize_speech,
                text=case["text"],
                output=f"test_output_{i}",
                streaming=False,
                **{k: v for k, v in case.items() if k != "text"}
            )
            for i, case in enumerate(test_cases)
        ]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result is not None)
            except Exception as e:
                print(f"Synthesis failed: {e}")
                results.append(False)
    
    total_time = time.time() - start_time
    successful_synthesis = sum(results)
    
    print(f"\nConcurrent Synthesis Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per request: {total_time/len(test_cases):.2f} seconds")

    print(f"Successful synthesis: {successful_synthesis}/{len(test_cases)}")
    
    assert successful_synthesis == len(test_cases)

if __name__ == "__main__":
    print("Running concurrent TTS request tests...")
    total_start_time = time.time()

    test_concurrent_synthesis()
    
    print(f"\nTotal Test Execution Time: {time.time() - total_start_time:.2f} seconds")
