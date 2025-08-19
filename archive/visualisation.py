import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class HealthDataVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Setup plotting style for better visualizations"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def plot_vital_signs_distribution(self):
        """Plot distribution of vital signs"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Vital Signs', fontsize=16, fontweight='bold')

        vital_signs = ['heart_rate', 'spo2', 'temperature', 'systolic_bp', 'diastolic_bp', 'age']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'lightcoral']

        for i, (vital, color) in enumerate(zip(vital_signs, colors)):
            row, col = divmod(i, 3)
            axes[row, col].hist(self.data[vital], bins=50, color=color, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{vital.replace("_", " ").title()} Distribution')
            axes[row, col].set_xlabel(vital.replace("_", " ").title())
            axes[row, col].set_ylabel('Frequency')

            # Add mean line
            mean_val = self.data[vital].mean()
            axes[row, col].axvline(mean_val, color='red', linestyle='--',
                                   label=f'Mean: {mean_val:.2f}')
            axes[row, col].legend()

        plt.tight_layout()
        plt.show()

    def plot_condition_prevalence(self):
        """Plot prevalence of different health conditions"""
        condition_counts = self.data[self.model.condition_names].sum()
        condition_percentages = (condition_counts / len(self.data)) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of counts
        bars1 = ax1.bar(range(len(condition_counts)), condition_counts.values,
                        color=plt.cm.Set3(np.linspace(0, 1, len(condition_counts))))
        ax1.set_title('Health Condition Counts', fontweight='bold')
        ax1.set_xlabel('Health Conditions')
        ax1.set_ylabel('Number of Cases')
        ax1.set_xticks(range(len(condition_counts)))
        ax1.set_xticklabels(condition_counts.index, rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars1, condition_counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(count), ha='center', va='bottom')

        # Pie chart of percentages
        wedges, texts, autotexts = ax2.pie(condition_percentages.values,
                                           labels=condition_percentages.index,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('Health Condition Prevalence (%)', fontweight='bold')

        plt.setp(autotexts, size=8, weight="bold")
        plt.setp(texts, size=8)

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """Plot correlation matrix between vital signs and conditions"""
        # Calculate correlations
        correlation_data = self.data[self.model.feature_names + self.model.condition_names]
        correlation_matrix = correlation_data.corr()

        # Create heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                    center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title('Correlation Matrix: Vital Signs and Health Conditions',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_model_performance(self, evaluation_results):
        """Plot model performance comparison"""
        models = list(evaluation_results.keys())
        overall_accuracies = [evaluation_results[model]['overall_accuracy']
                              for model in models]

        # Overall performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        bars = ax1.bar(models, overall_accuracies,
                       color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        ax1.set_title('Overall Model Performance', fontweight='bold')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, overall_accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Condition-specific performance for best model
        best_model = max(evaluation_results.keys(),
                         key=lambda x: evaluation_results[x]['overall_accuracy'])
        condition_accs = evaluation_results[best_model]['condition_accuracies']

        bars2 = ax2.bar(range(len(condition_accs)), list(condition_accs.values()),
                        color=plt.cm.viridis(np.linspace(0, 1, len(condition_accs))))
        ax2.set_title(f'Condition-Specific Performance ({best_model})', fontweight='bold')
        ax2.set_xlabel('Health Conditions')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(range(len(condition_accs)))
        ax2.set_xticklabels(list(condition_accs.keys()), rotation=45, ha='right')
        ax2.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars2, condition_accs.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

    def plot_vital_signs_by_condition(self, condition='fever'):
        """Plot vital signs distribution by health condition"""
        if condition not in self.model.condition_names:
            print(f"Condition '{condition}' not found in available conditions")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Vital Signs Distribution by {condition.title()} Status',
                     fontsize=16, fontweight='bold')

        vital_signs = ['heart_rate', 'spo2', 'temperature', 'systolic_bp']

        for i, vital in enumerate(vital_signs):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            # Separate data by condition
            positive = self.data[self.data[condition] == 1][vital]
            negative = self.data[self.data[condition] == 0][vital]

            # Plot distributions
            ax.hist(negative, bins=30, alpha=0.7, label=f'No {condition}',
                    color='lightblue', density=True)
            ax.hist(positive, bins=30, alpha=0.7, label=f'{condition.title()}',
                    color='salmon', density=True)

            ax.set_title(f'{vital.replace("_", " ").title()}')
            ax.set_xlabel(vital.replace("_", " ").title())
            ax.set_ylabel('Density')
            ax.legend()

            # Add mean lines
            ax.axvline(negative.mean(), color='blue', linestyle='--', alpha=0.8)
            ax.axvline(positive.mean(), color='red', linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.show()

    def create_interactive_dashboard(self):
        """Create an interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Heart Rate vs Age', 'SpO2 vs Temperature',
                            'Blood Pressure Relationship', 'Condition Prevalence'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )

        # Heart Rate vs Age (colored by gender)
        colors = ['red' if g == 1 else 'blue' for g in self.data['gender']]
        fig.add_trace(
            go.Scatter(x=self.data['age'], y=self.data['heart_rate'],
                       mode='markers', marker=dict(color=colors, opacity=0.6),
                       name='Heart Rate vs Age',
                       text=[f'Gender: {"Male" if g == 1 else "Female"}' for g in self.data['gender']]),
            row=1, col=1
        )

        # SpO2 vs Temperature
        fig.add_trace(
            go.Scatter(x=self.data['temperature'], y=self.data['spo2'],
                       mode='markers', marker=dict(color='green', opacity=0.6),
                       name='SpO2 vs Temperature'),
            row=1, col=2
        )

        # Blood Pressure Relationship
        fig.add_trace(
            go.Scatter(x=self.data['systolic_bp'], y=self.data['diastolic_bp'],
                       mode='markers', marker=dict(color='purple', opacity=0.6),
                       name='Systolic vs Diastolic BP'),
            row=2, col=1
        )

        # Condition Prevalence Pie Chart
        condition_counts = self.data[self.model.condition_names].sum()
        fig.add_trace(
            go.Pie(labels=condition_counts.index, values=condition_counts.values,
                   name="Condition Prevalence"),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(height=800, showlegend=True,
                          title_text="Health Monitoring Dashboard")

        fig.show()

    def plot_ecg_patterns(self):
        """Plot ECG-related measurements"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('ECG Pattern Analysis', fontsize=16, fontweight='bold')

        # RR Interval vs Heart Rate
        axes[0, 0].scatter(self.data['heart_rate'], self.data['ecg_rr_interval'],
                           alpha=0.6, color='blue')
        axes[0, 0].set_xlabel('Heart Rate (bpm)')
        axes[0, 0].set_ylabel('RR Interval (s)')
        axes[0, 0].set_title('RR Interval vs Heart Rate')

        # PR Interval Distribution by Arrhythmia
        arrhythmia_yes = self.data[self.data['arrhythmia'] == 1]['ecg_pr_interval']
        arrhythmia_no = self.data[self.data['arrhythmia'] == 0]['ecg_pr_interval']

        axes[0, 1].hist(arrhythmia_no, bins=30, alpha=0.7, label='No Arrhythmia',
                        color='lightgreen', density=True)
        axes[0, 1].hist(arrhythmia_yes, bins=30, alpha=0.7, label='Arrhythmia',
                        color='red', density=True)
        axes[0, 1].set_xlabel('PR Interval (s)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('PR Interval Distribution')
        axes[0, 1].legend()

        # QT Interval vs Heart Rate
        axes[1, 0].scatter(self.data['heart_rate'], self.data['ecg_qt_interval'],
                           alpha=0.6, color='orange')
        axes[1, 0].set_xlabel('Heart Rate (bpm)')
        axes[1, 0].set_ylabel('QT Interval (s)')
        axes[1, 0].set_title('QT Interval vs Heart Rate')

        # ST Elevation Distribution
        axes[1, 1].hist(self.data['ecg_st_elevation'], bins=50, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('ST Elevation (mm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('ST Elevation Distribution')
        axes[1, 1].axvline(2, color='red', linestyle='--', label='Abnormal Threshold')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def generate_patient_report(self, patient_data, predictions):
        """Generate a comprehensive patient report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PATIENT HEALTH REPORT")
        print("=" * 80)

        print(f"\nPATIENT DEMOGRAPHICS:")
        print(f"Age: {patient_data['age']} years")
        print(f"Gender: {'Male' if patient_data['gender'] == 1 else 'Female'}")

        print(f"\nVITAL SIGNS:")
        print(f"Heart Rate: {patient_data['heart_rate']} bpm")
        print(f"SpO2: {patient_data['spo2']}%")
        print(f"Temperature: {patient_data['temperature']}°C")
        print(f"Blood Pressure: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg")

        print(f"\nECG PARAMETERS:")
        print(f"RR Interval: {patient_data['ecg_rr_interval']:.3f} s")
        print(f"PR Interval: {patient_data['ecg_pr_interval']:.3f} s")
        print(f"QT Interval: {patient_data['ecg_qt_interval']:.3f} s")
        print(f"ST Elevation: {patient_data['ecg_st_elevation']:.1f} mm")

        print(f"\nRISK ASSESSMENT:")
        high_risk_conditions = []
        medium_risk_conditions = []
        low_risk_conditions = []

        for condition, result in predictions.items():
            if result['prediction']:
                if result['probability'] > 0.8:
                    high_risk_conditions.append((condition, result['probability']))
                elif result['probability'] > 0.5:
                    medium_risk_conditions.append((condition, result['probability']))
                else:
                    low_risk_conditions.append((condition, result['probability']))

        if high_risk_conditions:
            print(f"\n🚨 HIGH RISK CONDITIONS:")
            for condition, prob in high_risk_conditions:
                print(f"  ⚠️  {condition.upper()}: {prob:.1%} confidence")

        if medium_risk_conditions:
            print(f"\n⚠️  MEDIUM RISK CONDITIONS:")
            for condition, prob in medium_risk_conditions:
                print(f"  🔶 {condition.title()}: {prob:.1%} confidence")

        if low_risk_conditions:
            print(f"\n💛 LOW RISK CONDITIONS:")
            for condition, prob in low_risk_conditions:
                print(f"  🔸 {condition.title()}: {prob:.1%} confidence")

        if not (high_risk_conditions or medium_risk_conditions or low_risk_conditions):
            print(f"\n✅ NO SIGNIFICANT HEALTH RISKS DETECTED")

        print(f"\nRECOMMENDATIONS:")
        recommendations = self.model.get_health_recommendations(predictions)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        print("=" * 80)


# Complete example with all visualizations
def complete_health_analysis():
    """Complete health monitoring analysis with all visualizations"""
    from health_monitoring_model import HealthMonitoringModel

    print("🏥 COMPREHENSIVE HEALTH MONITORING SYSTEM")
    print("=" * 60)

    # Initialize the model
    health_model = HealthMonitoringModel()

    # Generate synthetic data
    print("\n📊 Generating synthetic health monitoring data...")
    df = health_model.generate_synthetic_data(n_samples=5000)

    # Preprocess and train
    print("🧠 Training machine learning models...")
    X, y = health_model.preprocess_data(df)
    history = health_model.train_models(X, y)

    # Evaluate models
    print("📈 Evaluating model performance...")
    results = health_model.evaluate_models()

    # Initialize visualizer
    visualizer = HealthDataVisualizer(health_model, df)

    # Create all visualizations
    print("\n🎨 Creating visualizations...")

    print("1. Vital Signs Distribution...")
    visualizer.plot_vital_signs_distribution()

    print("2. Health Condition Prevalence...")
    visualizer.plot_condition_prevalence()

    print("3. Correlation Analysis...")
    visualizer.plot_correlation_matrix()

    print("4. Model Performance Comparison...")
    visualizer.plot_model_performance(results)

    print("5. ECG Pattern Analysis...")
    visualizer.plot_ecg_patterns()

    print("6. Condition-specific Vital Signs Analysis...")
    for condition in ['fever', 'hypoxia', 'tachycardia', 'hypertension']:
        print(f"   Analyzing {condition}...")
        visualizer.plot_vital_signs_by_condition(condition)

    print("7. Interactive Dashboard...")
    visualizer.create_interactive_dashboard()

    # Test with multiple patient examples
    print("\n🧑‍⚕️ PATIENT CASE STUDIES")
    print("=" * 60)

    # Case 1: Healthy patient
    healthy_patient = {
        'heart_rate': 72,
        'spo2': 98,
        'temperature': 36.8,
        'ecg_rr_interval': 0.833,
        'ecg_pr_interval': 0.16,
        'ecg_qt_interval': 0.40,
        'ecg_st_elevation': 0.5,
        'age': 35,
        'gender': 0,
        'systolic_bp': 120,
        'diastolic_bp': 80
    }

    print("\nCASE 1: HEALTHY PATIENT")
    predictions_healthy = health_model.predict_health_status(healthy_patient)
    visualizer.generate_patient_report(healthy_patient, predictions_healthy)

    # Case 2: Critical patient
    critical_patient = {
        'heart_rate': 45,
        'spo2': 88,
        'temperature': 39.5,
        'ecg_rr_interval': 1.33,
        'ecg_pr_interval': 0.25,
        'ecg_qt_interval': 0.48,
        'ecg_st_elevation': 4.2,
        'age': 68,
        'gender': 1,
        'systolic_bp': 180,
        'diastolic_bp': 110
    }

    print("\nCASE 2: CRITICAL PATIENT")
    predictions_critical = health_model.predict_health_status(critical_patient)
    visualizer.generate_patient_report(critical_patient, predictions_critical)

    # Case 3: Moderate risk patient
    moderate_patient = {
        'heart_rate': 110,
        'spo2': 94,
        'temperature': 38.2,
        'ecg_rr_interval': 0.545,
        'ecg_pr_interval': 0.18,
        'ecg_qt_interval': 0.42,
        'ecg_st_elevation': 1.2,
        'age': 55,
        'gender': 0,
        'systolic_bp': 145,
        'diastolic_bp': 92
    }

    print("\nCASE 3: MODERATE RISK PATIENT")
    predictions_moderate = health_model.predict_health_status(moderate_patient)
    visualizer.generate_patient_report(moderate_patient, predictions_moderate)

    # Performance summary
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("=" * 60)

    best_model = max(results.keys(), key=lambda x: results[x]['overall_accuracy'])
    print(f"Best Performing Model: {best_model}")
    print(f"Overall Accuracy: {results[best_model]['overall_accuracy']:.3f}")

    print(f"\nCondition-Specific Performance (Best Model):")
    for condition, accuracy in results[best_model]['condition_accuracies'].items():
        print(f"  {condition.ljust(20)}: {accuracy:.3f}")

    # Feature importance analysis
    if best_model == 'random_forest':
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        # Get feature importance from random forest
        rf_model = health_model.models['random_forest']
        feature_importance = {}

        for i, condition in enumerate(health_model.condition_names):
            estimator = rf_model.estimators_[i]
            importances = estimator.feature_importances_
            feature_importance[condition] = dict(zip(health_model.feature_names, importances))

        # Plot feature importance
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Feature Importance by Condition (Random Forest)', fontsize=16, fontweight='bold')

        for i, condition in enumerate(health_model.condition_names):
            row, col = divmod(i, 3)
            importances = list(feature_importance[condition].values())
            features = list(feature_importance[condition].keys())

            # Sort by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = [features[idx] for idx in sorted_idx]
            sorted_importances = [importances[idx] for idx in sorted_idx]

            axes[row, col].bar(range(len(sorted_features)), sorted_importances,
                               color=plt.cm.viridis(np.linspace(0, 1, len(sorted_features))))
            axes[row, col].set_title(f'{condition.title()}')
            axes[row, col].set_xticks(range(len(sorted_features)))
            axes[row, col].set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=8)
            axes[row, col].set_ylabel('Importance')

        plt.tight_layout()
        plt.show()

    return health_model, df, results, visualizer


if __name__ == "__main__":
    model, data, evaluation_results, viz = complete_health_analysis()
