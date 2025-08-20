import unittest
import os
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, mock_open
import warnings
import sys

# Add benchmarking directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'benchmarking'))

from excel_generator import (
    parse_summary_file, 
    collect_summary_metrics, 
    create_pivot_table, 
    rank_values_in_columns,
    generate_results_excel,
    print_summary_stats
)


class TestExcelGenerator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_summary_content = """Model: TestModel1
Dataset: dms
Score_Type: cosine
ROC_AUC: 0.850000
Average_Precision: 0.750000
F1_Score: 0.680000
Threshold: 0.500000
"""
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_summary_file(self, filename, content):
        """Helper to create a test summary file."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath
    
    def test_parse_summary_file_valid(self):
        """Test parsing a valid summary file."""
        filepath = self.create_test_summary_file("test_summary.txt", self.sample_summary_content)
        
        result = parse_summary_file(filepath)
        
        expected = {
            'Model': 'TestModel1',
            'Dataset': 'dms',
            'Score_Type': 'cosine',
            'ROC_AUC': '0.850000',
            'Average_Precision': '0.750000',
            'F1_Score': '0.680000',
            'Threshold': '0.500000'
        }
        
        self.assertEqual(result, expected)
    
    def test_parse_summary_file_invalid_format(self):
        """Test parsing a file with invalid format."""
        invalid_content = "Invalid file content\nNo colons here\n"
        filepath = self.create_test_summary_file("invalid.txt", invalid_content)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_summary_file(filepath)
            
        self.assertEqual(result, {})
    
    def test_parse_summary_file_nonexistent(self):
        """Test parsing a non-existent file."""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.txt")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = parse_summary_file(nonexistent_path)
            
        self.assertEqual(result, {})
        self.assertTrue(len(w) > 0)
    
    def test_collect_summary_metrics_multiple_files(self):
        """Test collecting metrics from multiple summary files."""
        # Create multiple test files
        file1_content = """Model: Model1
Dataset: dms
Score_Type: cosine
ROC_AUC: 0.850000
Average_Precision: 0.750000
F1_Score: 0.680000
Threshold: 0.500000
"""
        
        file2_content = """Model: Model2
Dataset: sabdab_ep
Score_Type: seq_identity
ROC_AUC: 0.800000
Average_Precision: 0.700000
F1_Score: 0.650000
Threshold: 0.450000
"""
        
        self.create_test_summary_file("model1_dms_summarymetrics.txt", file1_content)
        self.create_test_summary_file("model2_sabdab_ep_summarymetrics.txt", file2_content)
        
        df = collect_summary_metrics(self.test_dir)
        
        self.assertEqual(len(df), 2)
        self.assertIn('Model1', df['Model'].values)
        self.assertIn('Model2', df['Model'].values)
        self.assertEqual(df['ROC_AUC'].dtype, 'float64')
        self.assertAlmostEqual(df[df['Model'] == 'Model1']['ROC_AUC'].iloc[0], 0.85, places=5)
    
    def test_collect_summary_metrics_no_files(self):
        """Test collecting metrics when no files exist."""
        with self.assertRaises(FileNotFoundError):
            collect_summary_metrics(self.test_dir)
    
    def test_collect_summary_metrics_invalid_files_only(self):
        """Test collecting metrics when only invalid files exist."""
        self.create_test_summary_file("invalid_summarymetrics.txt", "Invalid content")
        
        with self.assertRaises(ValueError):
            collect_summary_metrics(self.test_dir)
    
    def test_create_pivot_table(self):
        """Test creating a pivot table from collected metrics."""
        # Create test DataFrame
        data = [
            {'Model': 'Model1', 'Dataset': 'dms', 'ROC_AUC': 0.85, 'Average_Precision': 0.75, 'F1_Score': 0.68},
            {'Model': 'Model1', 'Dataset': 'sabdab_ep', 'ROC_AUC': 0.80, 'Average_Precision': 0.70, 'F1_Score': 0.65},
            {'Model': 'Model2', 'Dataset': 'dms', 'ROC_AUC': 0.78, 'Average_Precision': 0.72, 'F1_Score': 0.63},
        ]
        df = pd.DataFrame(data)
        
        pivot_df = create_pivot_table(df)
        
        # Check structure
        self.assertIsInstance(pivot_df.columns, pd.MultiIndex)
        self.assertIn('Model1', pivot_df.index)
        self.assertIn('Model2', pivot_df.index)
        
        # Check specific values
        self.assertAlmostEqual(pivot_df.loc['Model1', ('dms', 'ROC_AUC')], 0.85, places=5)
        self.assertAlmostEqual(pivot_df.loc['Model2', ('dms', 'F1_Score')], 0.63, places=5)
        
        # Check NaN for missing combinations
        self.assertTrue(pd.isna(pivot_df.loc['Model2', ('sabdab_ep', 'ROC_AUC')]))
    
    def test_rank_values_in_columns(self):
        """Test ranking values in pivot table columns."""
        # Create test pivot table
        models = ['Model1', 'Model2', 'Model3']
        columns = pd.MultiIndex.from_tuples([('dms', 'ROC_AUC'), ('dms', 'F1_Score')])
        
        data = [
            [0.85, 0.70],  # Model1
            [0.90, 0.65],  # Model2 - best ROC_AUC
            [0.80, 0.75],  # Model3 - best F1_Score
        ]
        
        df = pd.DataFrame(data, index=models, columns=columns)
        
        rankings = rank_values_in_columns(df)
        
        # Check ROC_AUC rankings
        roc_rankings = dict(rankings[('dms', 'ROC_AUC')])
        self.assertEqual(roc_rankings['Model2'], 1)  # Highest value
        self.assertEqual(roc_rankings['Model1'], 2)  # Second highest
        self.assertEqual(roc_rankings['Model3'], 3)  # Lowest
        
        # Check F1_Score rankings
        f1_rankings = dict(rankings[('dms', 'F1_Score')])
        self.assertEqual(f1_rankings['Model3'], 1)  # Highest value
        self.assertEqual(f1_rankings['Model1'], 2)  # Second highest
        self.assertEqual(f1_rankings['Model2'], 3)  # Lowest
    
    def test_rank_values_with_ties(self):
        """Test ranking when there are tied values."""
        models = ['Model1', 'Model2', 'Model3']
        columns = pd.MultiIndex.from_tuples([('dms', 'ROC_AUC')])
        
        data = [
            [0.85],  # Model1
            [0.85],  # Model2 - tied for first
            [0.80],  # Model3
        ]
        
        df = pd.DataFrame(data, index=models, columns=columns)
        rankings = rank_values_in_columns(df)
        
        roc_rankings = dict(rankings[('dms', 'ROC_AUC')])
        # Both Model1 and Model2 should have rank 1
        self.assertEqual(roc_rankings['Model1'], 1)
        self.assertEqual(roc_rankings['Model2'], 1)
        self.assertEqual(roc_rankings['Model3'], 3)  # Next rank after tie
    
    def test_rank_values_with_nan(self):
        """Test ranking when some values are NaN."""
        models = ['Model1', 'Model2', 'Model3']
        columns = pd.MultiIndex.from_tuples([('dms', 'ROC_AUC')])
        
        data = [
            [0.85],  # Model1
            [float('nan')],  # Model2 - missing value
            [0.80],  # Model3
        ]
        
        df = pd.DataFrame(data, index=models, columns=columns)
        rankings = rank_values_in_columns(df)
        
        roc_rankings = dict(rankings[('dms', 'ROC_AUC')])
        # Only non-NaN values should be ranked
        self.assertEqual(len(roc_rankings), 2)
        self.assertNotIn('Model2', roc_rankings)
        self.assertEqual(roc_rankings['Model1'], 1)
        self.assertEqual(roc_rankings['Model3'], 2)
    
    def test_generate_results_excel_integration(self):
        """Integration test for the main generate_results_excel function."""
        # Create multiple test summary files
        files_content = [
            ("Model1_dms_summarymetrics.txt", """Model: Model1
Dataset: dms
Score_Type: cosine
ROC_AUC: 0.850000
Average_Precision: 0.750000
F1_Score: 0.680000
Threshold: 0.500000
"""),
            ("Model1_sabdab_ep_summarymetrics.txt", """Model: Model1
Dataset: sabdab_ep
Score_Type: cosine
ROC_AUC: 0.800000
Average_Precision: 0.700000
F1_Score: 0.650000
Threshold: 0.450000
"""),
            ("Model2_dms_summarymetrics.txt", """Model: Model2
Dataset: dms
Score_Type: seq_identity
ROC_AUC: 0.780000
Average_Precision: 0.720000
F1_Score: 0.630000
Threshold: 0.480000
""")
        ]
        
        for filename, content in files_content:
            self.create_test_summary_file(filename, content)
        
        # Generate Excel file
        excel_path = generate_results_excel(self.test_dir, "test_results.xlsx")
        
        # Verify file was created
        self.assertTrue(os.path.exists(excel_path))
        self.assertTrue(excel_path.endswith("test_results.xlsx"))
        
        # Verify we can read the file (basic check)
        # Note: We could use openpyxl to read and verify content, but this tests the basic functionality
        file_size = os.path.getsize(excel_path)
        self.assertGreater(file_size, 0)
    
    def test_print_summary_stats(self):
        """Test the summary statistics printing function."""
        # Create test files
        self.create_test_summary_file("model1_dms_summarymetrics.txt", self.sample_summary_content)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            print_summary_stats(self.test_dir)
            
        # Verify print was called (indicating function ran without error)
        self.assertTrue(mock_print.called)
    
    def test_custom_pattern_matching(self):
        """Test using custom patterns for file matching."""
        # Create files with different patterns
        self.create_test_summary_file("custom_metrics.txt", self.sample_summary_content)
        self.create_test_summary_file("other_summarymetrics.txt", self.sample_summary_content)
        
        # Test default pattern (should only find summarymetrics.txt)
        df_default = collect_summary_metrics(self.test_dir)
        self.assertEqual(len(df_default), 1)
        
        # Test custom pattern (should find custom file)
        df_custom = collect_summary_metrics(self.test_dir, "custom_*.txt")
        self.assertEqual(len(df_custom), 1)


def run_excel_generator_tests():
    """
    Run all tests for the excel_generator module.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("🧪 Running Excel Generator Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestExcelGenerator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print(f"\n📊 Test Results:")
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_excel_generator_tests()
    exit(0 if success else 1)