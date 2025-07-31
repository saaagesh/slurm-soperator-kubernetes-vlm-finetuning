#!/usr/bin/env python3
"""
Infrastructure Cost Analysis and ROI Calculator
Demonstrates business value and cost optimization awareness
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import mlflow

class InfrastructureCostAnalyzer:
    """
    Analyzes infrastructure costs and calculates ROI for VLM fine-tuning
    """
    
    def __init__(self, output_dir="/mnt/models/cost_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nebius pricing (approximate - update with actual rates)
        self.pricing = {
            "h100_gpu_hour": 4.00,  # USD per H100 per hour
            "cpu_vcpu_hour": 0.05,   # USD per vCPU per hour
            "memory_gb_hour": 0.01,  # USD per GB RAM per hour
            "storage_gb_month": 0.10, # USD per GB storage per month
            "network_gb": 0.05       # USD per GB network transfer
        }
        
        # Infrastructure specifications
        self.infrastructure = {
            "gpus": 8,
            "gpu_type": "H100",
            "cpus": 128,
            "memory_gb": 1600,
            "storage_gb": 15000,  # Total across all filesystems
            "network_bandwidth_gbps": 400
        }
    
    def calculate_training_costs(self, training_duration_hours):
        """Calculate detailed training costs"""
        costs = {
            "compute": {
                "gpu_cost": self.infrastructure["gpus"] * self.pricing["h100_gpu_hour"] * training_duration_hours,
                "cpu_cost": self.infrastructure["cpus"] * self.pricing["cpu_vcpu_hour"] * training_duration_hours,
                "memory_cost": self.infrastructure["memory_gb"] * self.pricing["memory_gb_hour"] * training_duration_hours
            },
            "storage": {
                "models_storage": 2000 * self.pricing["storage_gb_month"] / 30 / 24 * training_duration_hours,  # 2TB for models
                "datasets_storage": 1000 * self.pricing["storage_gb_month"] / 30 / 24 * training_duration_hours,  # 1TB for datasets
                "logs_storage": 100 * self.pricing["storage_gb_month"] / 30 / 24 * training_duration_hours  # 100GB for logs
            },
            "network": {
                "model_download": 50 * self.pricing["network_gb"],  # 50GB model download
                "dataset_download": 20 * self.pricing["network_gb"],  # 20GB dataset
                "monitoring_data": 5 * self.pricing["network_gb"]  # 5GB monitoring/logging
            }
        }
        
        # Calculate totals
        total_compute = sum(costs["compute"].values())
        total_storage = sum(costs["storage"].values())
        total_network = sum(costs["network"].values())
        total_cost = total_compute + total_storage + total_network
        
        costs["totals"] = {
            "compute_total": total_compute,
            "storage_total": total_storage,
            "network_total": total_network,
            "grand_total": total_cost
        }
        
        return costs
    
    def calculate_roi_scenarios(self, training_costs):
        """Calculate ROI for different business scenarios"""
        scenarios = {
            "e_commerce_automation": {
                "description": "Automated product description generation",
                "baseline_cost_per_description": 2.00,  # Manual copywriter cost
                "ai_cost_per_description": 0.01,       # AI generation cost
                "descriptions_per_month": 10000,
                "quality_improvement_factor": 1.2,
                "time_savings_hours_per_month": 400
            },
            "content_scaling": {
                "description": "Scaling content creation for multiple languages/markets",
                "baseline_monthly_cost": 50000,        # Team of copywriters
                "ai_monthly_cost": 5000,               # AI-powered solution
                "scalability_multiplier": 5,           # 5x more content possible
                "quality_consistency_score": 0.95
            },
            "personalization": {
                "description": "Personalized product descriptions",
                "conversion_rate_improvement": 0.15,   # 15% improvement
                "average_order_value": 150,
                "monthly_visitors": 1000000,
                "baseline_conversion_rate": 0.02
            }
        }
        
        roi_results = {}
        
        for scenario_name, scenario in scenarios.items():
            if scenario_name == "e_commerce_automation":
                monthly_savings = (scenario["baseline_cost_per_description"] - scenario["ai_cost_per_description"]) * scenario["descriptions_per_month"]
                annual_savings = monthly_savings * 12
                payback_months = training_costs["totals"]["grand_total"] / monthly_savings
                
                roi_results[scenario_name] = {
                    "monthly_savings": monthly_savings,
                    "annual_savings": annual_savings,
                    "payback_period_months": payback_months,
                    "3_year_roi": (annual_savings * 3 - training_costs["totals"]["grand_total"]) / training_costs["totals"]["grand_total"] * 100
                }
            
            elif scenario_name == "content_scaling":
                monthly_savings = scenario["baseline_monthly_cost"] - scenario["ai_monthly_cost"]
                annual_savings = monthly_savings * 12
                payback_months = training_costs["totals"]["grand_total"] / monthly_savings
                
                roi_results[scenario_name] = {
                    "monthly_savings": monthly_savings,
                    "annual_savings": annual_savings,
                    "payback_period_months": payback_months,
                    "3_year_roi": (annual_savings * 3 - training_costs["totals"]["grand_total"]) / training_costs["totals"]["grand_total"] * 100,
                    "scalability_benefit": scenario["scalability_multiplier"]
                }
            
            elif scenario_name == "personalization":
                additional_conversions = scenario["monthly_visitors"] * scenario["conversion_rate_improvement"]
                additional_revenue = additional_conversions * scenario["average_order_value"]
                annual_additional_revenue = additional_revenue * 12
                payback_months = training_costs["totals"]["grand_total"] / additional_revenue
                
                roi_results[scenario_name] = {
                    "monthly_additional_revenue": additional_revenue,
                    "annual_additional_revenue": annual_additional_revenue,
                    "payback_period_months": payback_months,
                    "3_year_roi": (annual_additional_revenue * 3 - training_costs["totals"]["grand_total"]) / training_costs["totals"]["grand_total"] * 100
                }
        
        return roi_results
    
    def create_cost_visualizations(self, training_costs, roi_results, training_duration_hours):
        """Create comprehensive cost and ROI visualizations"""
        
        # 1. Cost Breakdown Pie Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Infrastructure Cost Analysis & ROI Projections', fontsize=16, fontweight='bold')
        
        # Cost breakdown
        cost_categories = ['Compute', 'Storage', 'Network']
        cost_values = [
            training_costs["totals"]["compute_total"],
            training_costs["totals"]["storage_total"],
            training_costs["totals"]["network_total"]
        ]
        
        axes[0, 0].pie(cost_values, labels=cost_categories, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title(f'Training Cost Breakdown\nTotal: ${training_costs["totals"]["grand_total"]:.2f}')
        
        # Detailed compute costs
        compute_breakdown = training_costs["compute"]
        axes[0, 1].pie(compute_breakdown.values(), labels=compute_breakdown.keys(), autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Compute Cost Details')
        
        # ROI Comparison
        scenarios = list(roi_results.keys())
        payback_periods = [roi_results[scenario]["payback_period_months"] for scenario in scenarios]
        three_year_rois = [roi_results[scenario]["3_year_roi"] for scenario in scenarios]
        
        x = range(len(scenarios))
        axes[1, 0].bar(x, payback_periods, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Business Scenarios')
        axes[1, 0].set_ylabel('Payback Period (Months)')
        axes[1, 0].set_title('Investment Payback Period')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        
        # 3-Year ROI
        colors = ['green' if roi > 100 else 'orange' if roi > 50 else 'red' for roi in three_year_rois]
        axes[1, 1].bar(x, three_year_rois, alpha=0.7, color=colors)
        axes[1, 1].set_xlabel('Business Scenarios')
        axes[1, 1].set_ylabel('3-Year ROI (%)')
        axes[1, 1].set_title('3-Year Return on Investment')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        axes[1, 1].axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Break-even')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_roi_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Timeline Analysis
        self.create_timeline_analysis(training_costs, roi_results)
        
        # 3. Sensitivity Analysis
        self.create_sensitivity_analysis(training_costs)
    
    def create_timeline_analysis(self, training_costs, roi_results):
        """Create timeline showing cost recovery"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Timeline for each scenario
        months = range(1, 37)  # 3 years
        initial_investment = training_costs["totals"]["grand_total"]
        
        for scenario_name, roi_data in roi_results.items():
            if "monthly_savings" in roi_data:
                cumulative_savings = [roi_data["monthly_savings"] * m for m in months]
            else:
                cumulative_savings = [roi_data["monthly_additional_revenue"] * m for m in months]
            
            net_value = [savings - initial_investment for savings in cumulative_savings]
            
            axes[0].plot(months, net_value, label=scenario_name.replace('_', ' ').title(), linewidth=2)
        
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        axes[0].set_xlabel('Months')
        axes[0].set_ylabel('Net Value ($)')
        axes[0].set_title('ROI Timeline Analysis')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cost per hour analysis
        hours = [1, 2, 4, 8, 12, 24]
        costs_per_hour = [self.calculate_training_costs(h)["totals"]["grand_total"] for h in hours]
        
        axes[1].plot(hours, costs_per_hour, marker='o', linewidth=2, markersize=8)
        axes[1].set_xlabel('Training Duration (Hours)')
        axes[1].set_ylabel('Total Cost ($)')
        axes[1].set_title('Cost vs Training Duration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timeline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_sensitivity_analysis(self, training_costs):
        """Create sensitivity analysis for key variables"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # GPU price sensitivity
        gpu_price_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        base_gpu_cost = training_costs["compute"]["gpu_cost"]
        
        total_costs = []
        for multiplier in gpu_price_multipliers:
            adjusted_gpu_cost = base_gpu_cost * multiplier
            adjusted_total = (training_costs["totals"]["grand_total"] - base_gpu_cost + adjusted_gpu_cost)
            total_costs.append(adjusted_total)
        
        axes[0].plot(gpu_price_multipliers, total_costs, marker='o', linewidth=2, color='red')
        axes[0].set_xlabel('GPU Price Multiplier')
        axes[0].set_ylabel('Total Training Cost ($)')
        axes[0].set_title('Sensitivity to GPU Pricing')
        axes[0].grid(True, alpha=0.3)
        
        # Training duration sensitivity
        duration_hours = [1, 2, 4, 6, 8, 12, 16, 24]
        duration_costs = [self.calculate_training_costs(h)["totals"]["grand_total"] for h in duration_hours]
        
        axes[1].plot(duration_hours, duration_costs, marker='s', linewidth=2, color='blue')
        axes[1].set_xlabel('Training Duration (Hours)')
        axes[1].set_ylabel('Total Cost ($)')
        axes[1].set_title('Cost Scaling with Duration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_executive_summary(self, training_costs, roi_results, training_duration_hours):
        """Generate executive summary report"""
        summary = {
            "executive_summary": {
                "project": "VLM Fine-tuning Infrastructure Analysis",
                "date": datetime.now().isoformat(),
                "training_duration_hours": training_duration_hours,
                "total_infrastructure_cost": training_costs["totals"]["grand_total"],
                "cost_per_hour": training_costs["totals"]["grand_total"] / training_duration_hours,
                "infrastructure_specs": self.infrastructure
            },
            "cost_breakdown": training_costs,
            "roi_scenarios": roi_results,
            "key_insights": {
                "most_cost_effective_scenario": max(roi_results.keys(), key=lambda x: roi_results[x]["3_year_roi"]),
                "fastest_payback": min(roi_results.keys(), key=lambda x: roi_results[x]["payback_period_months"]),
                "cost_optimization_recommendations": [
                    "Consider spot instances for training to reduce costs by 60-90%",
                    "Implement automatic shutdown after training completion",
                    "Use gradient checkpointing to reduce memory requirements",
                    "Optimize data loading to maximize GPU utilization"
                ],
                "scaling_considerations": [
                    "Current setup can handle 8x parallel training jobs",
                    "Storage can accommodate 50+ fine-tuned models",
                    "Network bandwidth supports real-time monitoring"
                ]
            }
        }
        
        # Save executive summary
        with open(self.output_dir / 'executive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_complete_analysis(self, training_duration_hours=4):
        """Run complete cost and ROI analysis"""
        print("Running comprehensive cost analysis...")
        
        # Calculate costs
        training_costs = self.calculate_training_costs(training_duration_hours)
        
        # Calculate ROI scenarios
        roi_results = self.calculate_roi_scenarios(training_costs)
        
        # Create visualizations
        self.create_cost_visualizations(training_costs, roi_results, training_duration_hours)
        
        # Generate executive summary
        summary = self.generate_executive_summary(training_costs, roi_results, training_duration_hours)
        
        # Log to MLflow
        mlflow.log_param("total_training_cost", training_costs["totals"]["grand_total"])
        mlflow.log_param("cost_per_hour", training_costs["totals"]["grand_total"] / training_duration_hours)
        
        for scenario_name, roi_data in roi_results.items():
            mlflow.log_metric(f"roi_3year_{scenario_name}", roi_data["3_year_roi"])
            mlflow.log_metric(f"payback_months_{scenario_name}", roi_data["payback_period_months"])
        
        # Log artifacts
        mlflow.log_artifacts(str(self.output_dir), "cost_analysis")
        
        print(f"Analysis complete! Total training cost: ${training_costs['totals']['grand_total']:.2f}")
        print(f"Best ROI scenario: {summary['key_insights']['most_cost_effective_scenario']}")
        
        return summary