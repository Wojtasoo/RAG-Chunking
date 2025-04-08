import pandas as pd
from pipeline import evaluate_pipeline

def run_experiments():
    corpora_dir = "../data/corpora"
    questions_csv = "../data/corpora/questions_df.csv"
    embedding_model = "all-MiniLM-L6-v2"
    
    experiments = [
        {"chunk_size": 200, "top_n": 5},
        {"chunk_size": 200, "top_n": 10},
        {"chunk_size": 400, "top_n": 5},
        {"chunk_size": 400, "top_n": 10}
    ]
    
    experiment_results = []
    cases= 0
    for exp in experiments:
        if cases<2:
            print(f"Running experiment with chunk_size={exp['chunk_size']} and top_n={exp['top_n']}")
            results_df = evaluate_pipeline(corpora_dir, questions_csv, exp["chunk_size"], exp["top_n"], embedding_model)
            results_df = pd.DataFrame(results_df)
            avg_precision = results_df["precision"].mean()
            avg_recall = results_df["recall"].mean()
            experiment_results.append({
                "chunk_size": exp["chunk_size"],
                "top_n": exp["top_n"],
                "avg_precision": avg_precision,
                "avg_recall": avg_recall
            })
            cases+=1
        else:
            break
        #individual experiment metrics if needed
        #results_df.to_csv(f"../data/metrics_chunk{exp['chunk_size']}_top{exp['top_n']}.csv", index=False)
    
    comparison_df = pd.DataFrame(experiment_results)
    print("\nComparison of Experiments:")
    print(comparison_df)
    comparison_df.to_csv("../data/experiment_comparison.csv", index=False)

if __name__ == "__main__":
    run_experiments()
