import os
import sys
import time

# Ensure package imports work regardless of execution location
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

# Verify that the Fireworks API key is configured or warn the user
api_key = os.getenv("FIREWORKS_API_KEY")
if not api_key:
    print("⚠️  Warning: FIREWORKS_API_KEY environment variable is not set.")
    print("Please set it in your environment or in a .env file before running.")
    print("e.g. export FIREWORKS_API_KEY='your-key-here'\n")

try:
    from cache.wrapper import SemanticCacheWrapper
    from cache.llm_evaluator import LLMEvaluator
    from cache.evals import PerfEval
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you have installed all dependencies in your environment.")
    sys.exit(1)


def run_test():
    print("   Initializing Semantic Cache Test Suite...")
    
    # 1. Initialize Semantic Cache Wrapper
    # By default, uses local Redis running on localhost:6379
    try:
        cache = SemanticCacheWrapper(
            distance_threshold=0.3,
            ttl=3600
        )
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        print("Please ensure Redis is running. You can start it using:")
        print("  docker run -d --name redis -p 6379:6379 redis/redis-stack:latest")
        return

    # 2. Configure LLM Evaluator using Fireworks minimax-m3
    model_name = "accounts/fireworks/models/minimax-m3"
    print(f" Setting up LLM Evaluator using model: {model_name}...")
    
    evaluator = LLMEvaluator.construct_with_fireworks(
        model=model_name
    )
    
    # Create and register the LLM-based Reranker
    reranker = evaluator.create_reranker(batch_size=5)
    cache.register_reranker(reranker)
    print("Registered Fireworks LLM Reranker.")

    # 3. Setup the Main LLM for cache misses (also using minimax-m3)
    print(f"  Setting up Main LLM using model: {model_name}...")
    main_llm = ChatOpenAI(
        openai_api_base="https://api.fireworks.ai/inference/v1",
        openai_api_key=api_key,
        model=model_name,
        temperature=0.7,
    )

    # 4. Seed the cache with FAQ data
    faq_data = [
        ("What is the return policy for purchases?", "You can return any purchase within 30 days for a full refund."),
        ("How do I track my order?", "Once shipped, you will receive an email with a tracking number and link."),
        ("Do you offer international shipping?", "Yes, we ship to over 100 countries worldwide. Rates vary by destination."),
        ("How can I contact customer support?", "You can reach customer support 24/7 via email at support@example.com or phone at 1-800-555-0199."),
    ]
    
    print("\nSeeding the cache with FAQ data...")
    cache.hydrate_from_pairs(faq_data, clear=False)
    print(f"Cache seeded with {len(faq_data)} entries.")

    # 5. Define test queries
    # Some should result in Cache Hits (semantically close to FAQs), others in Cache Misses (unrelated)
    test_queries = [
        # Expected Hits (similar to FAQs)
        ("Can I return something I bought?", True),
        ("Where is my package tracking info?", True),
        ("What is the cost of shipping to Japan?", True),
        ("What are your customer service contact details?", True),
        ("Can I return something I bought?", True),
        # Expected Misses (unrelated queries)
        ("What is the capital of France?", False),
        ("Tell me a programming joke.", False),
    ]

    # Initialize Performance and Cost Evaluator
    perf_eval = PerfEval()
    perf_eval.set_total_queries(len(test_queries))

    hits_count = 0
    misses_count = 0

    print("\nRunning test queries...")
    print("=" * 80)
    
    with perf_eval:
        for idx, (query, expected_hit) in enumerate(test_queries, 1):
            print(f"\n[{idx}/{len(test_queries)}] Query: '{query}'")
            print(f"Expected: {'Cache HIT' if expected_hit else 'Cache MISS'}")
            
            perf_eval.start()
            
            # Check the cache (with registered LLM reranker applied automatically)
            cache_results = cache.check(query)
            
            if cache_results.matches:
                # Cache HIT
                match = cache_results.matches[0]
                latency_label = "cache_hit"
                perf_eval.tick(latency_label)
                
                hits_count += 1
                print(f"   Actual: Cache HIT!")
                print(f"   Matched Prompt: '{match.prompt}'")
                print(f"   Response: '{match.response}'")
                print(f"   Reranker Score: {match.reranker_score}")
                print(f"   Reranker Reason: {match.reranker_reason}")
            else:
                # Cache MISS -> Call the Main LLM
                latency_label = "cache_miss"
                print(f"   Actual: Cache MISS!")
                
                if not api_key:
                    print("   ⚠️ Cannot invoke Fireworks API: FIREWORKS_API_KEY is not set.")
                    response_text = "Simulated response (API key missing)"
                    perf_eval.tick(latency_label)
                    misses_count += 1
                    continue
                
                try:
                    # Invoke the LLM
                    llm_start_time = time.time()
                    llm_response = main_llm.invoke(query)
                    response_text = llm_response.content
                    
                    # Record the LLM call for cost and token tracking
                    perf_eval.record_llm_call(
                        model=model_name,
                        input_text=query,
                        output_text=response_text,
                        provider="fireworks"
                    )
                    
                    # Store response in cache for future hits
                    cache.cache.store(prompt=query, response=response_text)
                    
                    # Tick the performance evaluator
                    perf_eval.tick(latency_label)
                    misses_count += 1
                    
                    print(f"   Generated Response: '{response_text}'")
                    print(f"   (Stored in cache for future queries)")
                except Exception as ex:
                    perf_eval.tick(latency_label)
                    misses_count += 1
                    print(f"   Error calling main LLM: {ex}")
            
            print("-" * 80)

    # 6. Show results and cost summaries
    print("\n===TEST RESULTS SUMMARY===")
    print("=" * 40)
    print(f"Total Queries: {len(test_queries)}")
    print(f"Cache Hits:    {hits_count}")
    print(f"Cache Misses:  {misses_count}")
    print(f"Hit Rate:      {(hits_count / len(test_queries)) * 100:.1f}%")
    
    # Print latencies and costs
    metrics = perf_eval.get_metrics(labels=["cache_hit", "cache_miss"])
    print(f"\n Latency Analysis:")
    print(f"  Overall Avg Latency: {metrics['overall']['average_latency']:.1f}ms")
    
    hit_metrics = metrics['by_label'].get('cache_hit', {})
    if hit_metrics.get('count', 0) > 0:
        print(f"  Cache Hit Avg Latency: {hit_metrics['average_latency']:.1f}ms")
        
    miss_metrics = metrics['by_label'].get('cache_miss', {})
    if miss_metrics.get('count', 0) > 0:
        print(f"  Cache Miss Avg Latency: {miss_metrics['average_latency']:.1f}ms")

    costs = perf_eval.get_costs()
    print(f"\n   Cost Analysis (Model: {model_name}):")
    print(f"  Total LLM Calls:      {costs['calls']}")
    print(f"  Total Cost:           ${costs['total_cost']:.6f}")
    print(f"  Avg Cost per Query:   ${costs.get('avg_cost_per_query', 0.0):.6f}")
    print("=" * 40)


if __name__ == "__main__":
    run_test()
