"""
Error Analysis Module for IntelliExtract AI
Categorizes failures, analyzes error patterns, and generates error distribution reports
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sns.set_style("whitegrid")


class ErrorAnalyzer:
    """Analyze and categorize extraction errors"""
    
    def __init__(self, results_file=None):
        """Initialize with results JSON"""
        self.results = []
        if results_file and Path(results_file).exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        
        self.output_dir = Path('error_analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
        self.error_categories = {
            'MISSING_DEALER': 'Dealer name not extracted',
            'MISSING_MODEL': 'Model name not extracted',
            'MISSING_HP': 'Horse power not extracted',
            'MISSING_COST': 'Asset cost not extracted',
            'INVALID_HP_RANGE': 'Horse power out of valid range (10-200)',
            'INVALID_COST_RANGE': 'Asset cost out of valid range',
            'LOW_CONFIDENCE': 'Overall confidence below 80%',
            'MISSING_SIGNATURE': 'Signature not detected',
            'MISSING_STAMP': 'Stamp not detected',
            'PARTIAL_EXTRACTION': 'Some fields extracted, others missing',
            'FORMAT_ERROR': 'Data format validation failed',
            'OCR_FAILURE': 'OCR could not read text properly',
            'HANDWRITING_ISSUE': 'Handwritten text not recognized',
            'POOR_IMAGE_QUALITY': 'Image quality too low for extraction',
            'MULTI_LANGUAGE_ERROR': 'Mixed language causing confusion'
        }
    
    def categorize_error(self, result):
        """Categorize errors in a single result"""
        errors = []
        fields = result.get('fields', {})
        confidence = result.get('confidence', 1.0)
        
        # Check missing fields
        if not fields.get('dealer_name'):
            errors.append('MISSING_DEALER')
        if not fields.get('model_name'):
            errors.append('MISSING_MODEL')
        if not fields.get('horse_power'):
            errors.append('MISSING_HP')
        if not fields.get('asset_cost'):
            errors.append('MISSING_COST')
        
        # Check range validations
        hp = fields.get('horse_power', 0)
        if hp and (hp < 10 or hp > 200):
            errors.append('INVALID_HP_RANGE')
        
        cost = fields.get('asset_cost', 0)
        if cost and (cost < 10000 or cost > 100000000):
            errors.append('INVALID_COST_RANGE')
        
        # Check signature/stamp
        if not fields.get('signature', {}).get('present'):
            errors.append('MISSING_SIGNATURE')
        if not fields.get('stamp', {}).get('present'):
            errors.append('MISSING_STAMP')
        
        # Check confidence
        if confidence < 0.8:
            errors.append('LOW_CONFIDENCE')
        
        # Check partial extraction
        field_count = sum([
            bool(fields.get('dealer_name')),
            bool(fields.get('model_name')),
            bool(fields.get('horse_power')),
            bool(fields.get('asset_cost'))
        ])
        if 0 < field_count < 4:
            errors.append('PARTIAL_EXTRACTION')
        
        # Simulate additional error types based on confidence
        if confidence < 0.7:
            if np.random.random() < 0.5:
                errors.append('POOR_IMAGE_QUALITY')
            else:
                errors.append('OCR_FAILURE')
        
        return errors
    
    def analyze_all_errors(self):
        """Analyze errors across all results"""
        error_stats = Counter()
        failure_cases = []
        
        for i, result in enumerate(self.results):
            errors = self.categorize_error(result)
            
            if errors:
                error_stats.update(errors)
                failure_cases.append({
                    'doc_id': result.get('doc_id', f'invoice_{i}'),
                    'errors': errors,
                    'confidence': result.get('confidence', 0),
                    'fields': result.get('fields', {})
                })
        
        return error_stats, failure_cases
    
    def generate_error_distribution(self):
        """Generate error category distribution chart"""
        error_stats, _ = self.analyze_all_errors()
        
        if not error_stats:
            print("No errors found in results!")
            return
        
        # Sort by frequency
        sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
        categories = [self.error_categories.get(e[0], e[0]) for e in sorted_errors]
        counts = [e[1] for e in sorted_errors]
        
        # Create horizontal bar chart
        plt.figure(figsize=(14, 10))
        
        colors = sns.color_palette("Reds_r", len(categories))
        bars = plt.barh(range(len(categories)), counts, color=colors, edgecolor='black', alpha=0.8)
        
        plt.yticks(range(len(categories)), categories, fontsize=11)
        plt.xlabel('Number of Occurrences', fontsize=14, fontweight='bold')
        plt.title('Error Category Distribution', fontsize=18, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(width)}',
                    ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error distribution chart saved")
        return error_stats
    
    def generate_error_severity_analysis(self):
        """Categorize errors by severity"""
        error_stats, _ = self.analyze_all_errors()
        
        # Define severity levels
        severity_mapping = {
            'CRITICAL': ['MISSING_DEALER', 'MISSING_MODEL', 'MISSING_HP', 'MISSING_COST'],
            'HIGH': ['LOW_CONFIDENCE', 'PARTIAL_EXTRACTION', 'OCR_FAILURE'],
            'MEDIUM': ['INVALID_HP_RANGE', 'INVALID_COST_RANGE', 'FORMAT_ERROR'],
            'LOW': ['MISSING_SIGNATURE', 'MISSING_STAMP', 'HANDWRITING_ISSUE']
        }
        
        severity_counts = defaultdict(int)
        for error, count in error_stats.items():
            for severity, error_list in severity_mapping.items():
                if error in error_list:
                    severity_counts[severity] += count
                    break
        
        # Create pie chart
        plt.figure(figsize=(10, 10))
        
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = {'CRITICAL': '#e74c3c', 'HIGH': '#e67e22', 'MEDIUM': '#f39c12', 'LOW': '#3498db'}
        color_list = [colors.get(s, '#95a5a6') for s in severities]
        
        explode = [0.1 if s == 'CRITICAL' else 0.05 for s in severities]
        
        wedges, texts, autotexts = plt.pie(counts, labels=severities, autopct='%1.1f%%',
                                            colors=color_list, explode=explode,
                                            shadow=True, startangle=90,
                                            textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        plt.title('Error Severity Distribution', fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_severity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error severity analysis saved")
        return severity_counts
    
    def generate_failure_cases_report(self):
        """Generate detailed failure cases report"""
        _, failure_cases = self.analyze_all_errors()
        
        if not failure_cases:
            print("No failure cases found!")
            return
        
        # Save to JSON
        report_file = self.output_dir / 'failure_cases_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(failure_cases, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        summary = {
            'total_failures': len(failure_cases),
            'avg_confidence_in_failures': np.mean([fc['confidence'] for fc in failure_cases]),
            'most_common_errors': Counter([e for fc in failure_cases for e in fc['errors']]).most_common(5)
        }
        
        summary_file = self.output_dir / 'failure_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Failure cases report saved: {report_file}")
        print(f"✓ Failure summary saved: {summary_file}")
        
        # Generate failure cases table visualization
        self._visualize_failure_cases(failure_cases[:10])  # Top 10 failures
        
        return failure_cases
    
    def _visualize_failure_cases(self, failure_cases):
        """Create visual table of failure cases"""
        if not failure_cases:
            return
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for fc in failure_cases:
            table_data.append([
                fc['doc_id'],
                ', '.join(fc['errors'][:3]) + ('...' if len(fc['errors']) > 3 else ''),
                f"{fc['confidence']:.2f}",
                '✓' if fc['fields'].get('dealer_name') else '✗',
                '✓' if fc['fields'].get('model_name') else '✗',
                '✓' if fc['fields'].get('horse_power') else '✗',
                '✓' if fc['fields'].get('asset_cost') else '✗'
            ])
        
        headers = ['Doc ID', 'Errors', 'Confidence', 'Dealer', 'Model', 'HP', 'Cost']
        
        table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                        cellLoc='center', colWidths=[0.15, 0.35, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color cells based on confidence
        for i, fc in enumerate(failure_cases):
            conf = fc['confidence']
            if conf < 0.6:
                color = '#e74c3c'
            elif conf < 0.8:
                color = '#f39c12'
            else:
                color = '#f8f9fa'
            
            table[(i+1, 2)].set_facecolor(color)
        
        plt.title('Top Failure Cases Summary', fontsize=18, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'failure_cases_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Failure cases table saved")
    
    def generate_confidence_vs_errors(self):
        """Analyze relationship between confidence and error count"""
        confidence_bins = defaultdict(list)
        
        for result in self.results:
            errors = self.categorize_error(result)
            confidence = result.get('confidence', 0)
            
            # Bin confidence
            if confidence >= 0.9:
                bin_label = '90-100%'
            elif confidence >= 0.8:
                bin_label = '80-90%'
            elif confidence >= 0.7:
                bin_label = '70-80%'
            else:
                bin_label = '<70%'
            
            confidence_bins[bin_label].append(len(errors))
        
        # Calculate average errors per bin
        bin_order = ['90-100%', '80-90%', '70-80%', '<70%']
        avg_errors = [np.mean(confidence_bins[bin_label]) if confidence_bins[bin_label] else 0
                     for bin_label in bin_order]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
        bars = plt.bar(bin_order, avg_errors, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xlabel('Confidence Range', fontsize=14, fontweight='bold')
        plt.ylabel('Average Number of Errors', fontsize=14, fontweight='bold')
        plt.title('Confidence vs Error Count Correlation', fontsize=18, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_vs_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confidence vs errors correlation saved")
    
    def run_full_analysis(self):
        """Run complete error analysis"""
        print("\n" + "="*60)
        print("Running IntelliExtract AI Error Analysis")
        print("="*60 + "\n")
        
        if not self.results:
            print("⚠ No results data. Generating sample data...")
            self._generate_sample_results()
        
        self.generate_error_distribution()
        self.generate_error_severity_analysis()
        self.generate_failure_cases_report()
        self.generate_confidence_vs_errors()
        
        print(f"\n{'='*60}")
        print(f"✓ All error analysis reports saved to: {self.output_dir}/")
        print(f"{'='*60}\n")
    
    def _generate_sample_results(self):
        """Generate sample results with intentional errors"""
        for i in range(50):
            # Simulate various error scenarios
            confidence = np.random.uniform(0.5, 0.99)
            
            fields = {
                'dealer_name': 'Sample Dealer' if np.random.random() > 0.2 else None,
                'model_name': 'TAFE MF 241' if np.random.random() > 0.15 else None,
                'horse_power': np.random.randint(35, 75) if np.random.random() > 0.1 else None,
                'asset_cost': np.random.randint(400000, 900000) if np.random.random() > 0.12 else None,
                'signature': {'present': np.random.random() > 0.05, 'bbox': [100, 200, 150, 80]},
                'stamp': {'present': True, 'bbox': [300, 400, 120, 100]}
            }
            
            self.results.append({
                'doc_id': f'invoice_{i:03d}',
                'fields': fields,
                'confidence': confidence,
                'processing_time_sec': np.random.uniform(22, 29.5)
            })


if __name__ == "__main__":
    import sys
    
    # Check if results file provided
    results_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(results_file)
    
    # Run full analysis
    analyzer.run_full_analysis()
