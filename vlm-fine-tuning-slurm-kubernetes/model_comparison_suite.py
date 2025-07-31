#!/usr/bin/env python3
"""
Comprehensive Model Comparison Suite with Automated Evaluation
Creates impressive visualizations and metrics for presentation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import json
import time
from pathlib import Path
import mlflow
import mlflow.pytorch
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from datasets import load_dataset
import re
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VLMModelComparator:
    """
    Comprehensive VLM model comparison with automated evaluation metrics
    """
    
    def __init__(self, base_model_path, finetuned_model_path, output_dir="/mnt/models/comparison"):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        self.base_model = None
        self.finetuned_model = None
        self.processor = None
        
        # Results storage
        self.comparison_results = {}
        self.evaluation_metrics = {}
        
    def load_models(self):
        """Load both base and fine-tuned models"""
        print("Loading base model...")
        self.base_model = MllamaForConditionalGeneration.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("Loading fine-tuned model...")
        self.finetuned_model = MllamaForConditionalGeneration.from_pretrained(
            self.finetuned_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("Loading processor...")
        self.processor = MllamaProcessor.from_pretrained(self.base_model_path)
        
    def evaluate_on_test_set(self, test_size=50):
        """Comprehensive evaluation on test dataset"""
        print(f"Evaluating on {test_size} test samples...")
        
        # Load test dataset
        dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
        test_data = dataset.shuffle(seed=42).select(range(test_size))
        
        results = {
            "base_model": [],
            "finetuned_model": [],
            "ground_truth": [],
            "images": [],
            "product_names": [],
            "categories": []
        }
        
        for i, sample in enumerate(test_data):
            print(f"Processing sample {i+1}/{test_size}")
            
            # Prepare input
            image = sample["image"].convert("RGB")
            product_name = sample["Product Name"]
            category = sample["Category"]
            ground_truth = sample["description"]
            
            prompt = f"""Based on the product image and the following information:
- Product Name: {product_name}
- Category: {category}

Please write a detailed and engaging product description that highlights the key features, benefits, and appeal of this product."""
            
            # Generate with base model
            base_output = self.generate_description(self.base_model, image, prompt)
            
            # Generate with fine-tuned model
            finetuned_output = self.generate_description(self.finetuned_model, image, prompt)
            
            # Store results
            results["base_model"].append(base_output)
            results["finetuned_model"].append(finetuned_output)
            results["ground_truth"].append(ground_truth)
            results["images"].append(image)
            results["product_names"].append(product_name)
            results["categories"].append(category)
        
        return results
    
    def generate_description(self, model, image, prompt):
        """Generate description using specified model"""
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        prompt_length = len(input_text)
        generated_description = generated_text[prompt_length:].strip()
        
        return generated_description
    
    def calculate_semantic_similarity(self, results):
        """Calculate semantic similarity using sentence transformers"""
        print("Calculating semantic similarity scores...")
        
        # Load sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode all descriptions
        ground_truth_embeddings = sentence_model.encode(results["ground_truth"])
        base_embeddings = sentence_model.encode(results["base_model"])
        finetuned_embeddings = sentence_model.encode(results["finetuned_model"])
        
        # Calculate similarities
        base_similarities = []
        finetuned_similarities = []
        
        for i in range(len(results["ground_truth"])):
            base_sim = cosine_similarity(
                [ground_truth_embeddings[i]], 
                [base_embeddings[i]]
            )[0][0]
            
            finetuned_sim = cosine_similarity(
                [ground_truth_embeddings[i]], 
                [finetuned_embeddings[i]]
            )[0][0]
            
            base_similarities.append(base_sim)
            finetuned_similarities.append(finetuned_sim)
        
        return base_similarities, finetuned_similarities
    
    def calculate_text_metrics(self, results):
        """Calculate various text quality metrics"""
        print("Calculating text quality metrics...")
        
        metrics = {
            "base_model": {
                "avg_length": np.mean([len(text.split()) for text in results["base_model"]]),
                "avg_sentences": np.mean([len(re.split(r'[.!?]+', text)) for text in results["base_model"]]),
                "unique_words": len(set(" ".join(results["base_model"]).lower().split())),
            },
            "finetuned_model": {
                "avg_length": np.mean([len(text.split()) for text in results["finetuned_model"]]),
                "avg_sentences": np.mean([len(re.split(r'[.!?]+', text)) for text in results["finetuned_model"]]),
                "unique_words": len(set(" ".join(results["finetuned_model"]).lower().split())),
            },
            "ground_truth": {
                "avg_length": np.mean([len(text.split()) for text in results["ground_truth"]]),
                "avg_sentences": np.mean([len(re.split(r'[.!?]+', text)) for text in results["ground_truth"]]),
                "unique_words": len(set(" ".join(results["ground_truth"]).lower().split())),
            }
        }
        
        return metrics
    
    def create_comparison_visualizations(self, results, base_similarities, finetuned_similarities, text_metrics):
        """Create comprehensive comparison visualizations"""
        print("Creating comparison visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Semantic Similarity Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VLM Model Comparison: Base vs Fine-tuned', fontsize=16, fontweight='bold')
        
        # Similarity distribution
        axes[0, 0].hist(base_similarities, alpha=0.7, label='Base Model', bins=20)
        axes[0, 0].hist(finetuned_similarities, alpha=0.7, label='Fine-tuned Model', bins=20)
        axes[0, 0].set_xlabel('Semantic Similarity to Ground Truth')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Semantic Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Improvement visualization
        improvements = np.array(finetuned_similarities) - np.array(base_similarities)
        axes[0, 1].bar(range(len(improvements)), improvements, 
                      color=['green' if x > 0 else 'red' for x in improvements])
        axes[0, 1].set_xlabel('Test Sample')
        axes[0, 1].set_ylabel('Similarity Improvement')
        axes[0, 1].set_title('Per-Sample Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Text length comparison
        lengths_data = pd.DataFrame({
            'Base Model': [len(text.split()) for text in results["base_model"]],
            'Fine-tuned Model': [len(text.split()) for text in results["finetuned_model"]],
            'Ground Truth': [len(text.split()) for text in results["ground_truth"]]
        })
        
        lengths_data.boxplot(ax=axes[1, 0])
        axes[1, 0].set_title('Description Length Distribution')
        axes[1, 0].set_ylabel('Word Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary metrics
        summary_metrics = {
            'Avg Similarity': [np.mean(base_similarities), np.mean(finetuned_similarities)],
            'Std Similarity': [np.std(base_similarities), np.std(finetuned_similarities)],
            'Max Similarity': [np.max(base_similarities), np.max(finetuned_similarities)]
        }
        
        x = np.arange(len(summary_metrics))
        width = 0.35
        
        for i, (metric, values) in enumerate(summary_metrics.items()):
            axes[1, 1].bar(x[i] - width/2, values[0], width, label='Base Model' if i == 0 else "")
            axes[1, 1].bar(x[i] + width/2, values[1], width, label='Fine-tuned Model' if i == 0 else "")
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Summary Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(summary_metrics.keys())
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Performance Matrix
        self.create_performance_heatmap(base_similarities, finetuned_similarities, results)
        
        # 3. Sample Comparison Grid
        self.create_sample_comparison_grid(results, base_similarities, finetuned_similarities)
    
    def create_performance_heatmap(self, base_similarities, finetuned_similarities, results):
        """Create performance heatmap by category"""
        # Group by category
        categories = list(set(results["categories"]))
        category_performance = []
        
        for category in categories:
            cat_indices = [i for i, cat in enumerate(results["categories"]) if cat == category]
            
            if cat_indices:
                base_avg = np.mean([base_similarities[i] for i in cat_indices])
                finetuned_avg = np.mean([finetuned_similarities[i] for i in cat_indices])
                improvement = finetuned_avg - base_avg
                
                category_performance.append([base_avg, finetuned_avg, improvement])
        
        # Create heatmap
        performance_df = pd.DataFrame(
            category_performance,
            columns=['Base Model', 'Fine-tuned Model', 'Improvement'],
            index=categories
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_df, annot=True, cmap='RdYlGn', center=0, 
                   fmt='.3f', cbar_kws={'label': 'Semantic Similarity'})
        plt.title('Performance by Product Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sample_comparison_grid(self, results, base_similarities, finetuned_similarities, num_samples=6):
        """Create a grid showing best improvements"""
        # Find best improvements
        improvements = np.array(finetuned_similarities) - np.array(base_similarities)
        best_indices = np.argsort(improvements)[-num_samples:]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Best Fine-tuning Improvements (Sample Outputs)', fontsize=16, fontweight='bold')
        
        for idx, sample_idx in enumerate(best_indices):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # Display image
            ax.imshow(results["images"][sample_idx])
            ax.axis('off')
            
            # Add text information
            product_name = results["product_names"][sample_idx][:30] + "..."
            improvement = improvements[sample_idx]
            
            ax.set_title(f'{product_name}\nImprovement: +{improvement:.3f}', 
                        fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_improvements_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_evaluation(self, test_size=50):
        """Run complete evaluation pipeline"""
        print("Starting comprehensive model evaluation...")
        
        # Load models
        self.load_models()
        
        # Evaluate on test set
        results = self.evaluate_on_test_set(test_size)
        
        # Calculate metrics
        base_similarities, finetuned_similarities = self.calculate_semantic_similarity(results)
        text_metrics = self.calculate_text_metrics(results)
        
        # Create visualizations
        self.create_comparison_visualizations(results, base_similarities, finetuned_similarities, text_metrics)
        
        # Calculate final metrics
        final_metrics = {
            "base_avg_similarity": float(np.mean(base_similarities)),
            "finetuned_avg_similarity": float(np.mean(finetuned_similarities)),
            "avg_improvement": float(np.mean(finetuned_similarities) - np.mean(base_similarities)),
            "improvement_percentage": float((np.mean(finetuned_similarities) - np.mean(base_similarities)) / np.mean(base_similarities) * 100),
            "samples_improved": int(sum(1 for i in range(len(base_similarities)) if finetuned_similarities[i] > base_similarities[i])),
            "improvement_rate": float(sum(1 for i in range(len(base_similarities)) if finetuned_similarities[i] > base_similarities[i]) / len(base_similarities) * 100)
        }
        
        # Save results
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                "final_metrics": final_metrics,
                "text_metrics": text_metrics,
                "detailed_results": {
                    "base_similarities": base_similarities,
                    "finetuned_similarities": finetuned_similarities,
                    "sample_count": len(results["base_model"])
                }
            }, f, indent=2)
        
        print("Evaluation completed!")
        print(f"Average similarity improvement: {final_metrics['avg_improvement']:.4f}")
        print(f"Improvement percentage: {final_metrics['improvement_percentage']:.2f}%")
        print(f"Samples improved: {final_metrics['samples_improved']}/{len(base_similarities)} ({final_metrics['improvement_rate']:.1f}%)")
        
        return final_metrics, results

# Usage in your workflow
def run_model_comparison():
    """Run comprehensive model comparison"""
    comparator = VLMModelComparator(
        base_model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
        finetuned_model_path="/mnt/models/fine-tuned"
    )
    
    metrics, results = comparator.run_comprehensive_evaluation(test_size=100)
    
    # Log to MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"evaluation_{metric_name}", value)
    
    # Log artifacts
    mlflow.log_artifacts(str(comparator.output_dir), "evaluation_results")
    
    return metrics, results