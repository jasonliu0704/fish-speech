import concurrent.futures
import time
from typing import List, Dict, Any
import pytest

from test_client import create_tts_request, synthesize_speech
from fish_speech.utils.schema import ServeTTSRequest

def generate_test_cases() -> List[Dict[str, Any]]:
    return [
        {
            "text": "This is test case 1",
            "normalize": True,
            "format": "wav",
            "temperature": 0.7
        },
        {
            "text": "This is test case 2 with different parameters",
            "normalize": False,
            "format": "wav",
            "temperature": 0.8
        },
        {
            "text": "Test case 3 with reference",
            "reference_id": "test_reference",
            "format": "wav",
            "temperature": 0.6
        }
    ]

def test_create_tts_request():
    test_cases = generate_test_cases()
    
    # Test single request creation
    request = create_tts_request(**test_cases[0])
    assert isinstance(request, ServeTTSRequest)
    assert request.text == test_cases[0]["text"]

def execute_tts_request(params: Dict[str, Any]) -> tuple[float, bool]:
    start_time = time.time()
    try:
        request = create_tts_request(**params)
        assert isinstance(request, ServeTTSRequest)
        success = True
    except Exception as e:
        print(f"Error processing request: {e}")
        success = False
    duration = time.time() - start_time
    return duration, success

def test_concurrent_requests():
    test_cases = generate_test_cases() * 3  # Create 9 test cases
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_params = {
            executor.submit(execute_tts_request, params): params 
            for params in test_cases
        }
        
        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                duration, success = future.result()
                results.append({
                    "params": params,
                    "duration": duration,
                    "success": success
                })
            except Exception as e:
                print(f"Request failed: {e}")
    
    total_time = time.time() - start_time
    successful_requests = sum(1 for r in results if r["success"])
    
    print(f"\nConcurrent Test Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {successful_requests}/{len(test_cases)}")
    print(f"Average request time: {sum(r['duration'] for r in results)/len(results):.2f} seconds")
    
    assert successful_requests == len(test_cases)

def test_concurrent_synthesis():
    test_cases = generate_test_cases()[:2]  # Use fewer cases for synthesis test
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
    print(f"Successful synthesis: {successful_synthesis}/{len(test_cases)}")
    
    assert successful_synthesis == len(test_cases)

if __name__ == "__main__":
    print("Running concurrent TTS request tests...")
    test_create_tts_request()
    test_concurrent_requests()
    test_concurrent_synthesis()
