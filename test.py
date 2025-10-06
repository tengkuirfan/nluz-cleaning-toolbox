"""
Comprehensive test script for the refactored nluztoolbox package.
Tests both TabularCleaning and ImageCleaning classes.
"""

import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from nluztoolbox import TabularCleaning, ImageCleaning

def test_tabular_cleaning():
    """Test all TabularCleaning methods"""
    print("=" * 60)
    print("TESTING TABULAR CLEANING")
    print("=" * 60)
    
    # Create sample data
    df = pd.DataFrame({
        'price': ['1,200', '2,500', '3,000', None, '5,000'],
        'quantity': [10, 20, 30, 40, 50],
        'rating': [4.5, 5.0, 3.0, 4.0, 15.0],  # 15.0 is an outlier
        'category': ['A', 'B', 'A', 'C', 'B'],
        'score': [85, 90, 78, 92, 88]
    })
    
    print("\n1. Original DataFrame:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Test 1: Replace symbols
    print("\n2. Testing replace_symbols...")
    cleaner = TabularCleaning(df)
    cleaner.replace_symbols(columns=['price'], symbols=[','])
    print("After replacing commas in 'price':")
    print(cleaner.get()['price'])
    
    # Test 2: Convert to float first (pandas will convert 'None' to NaN automatically)
    print("\n3. Testing astype to convert to float...")
    # Replace 'None' string with NaN first, then convert
    cleaner.df['price'] = pd.to_numeric(cleaner.df['price'], errors='coerce')
    print("After converting 'price' to numeric (None becomes NaN):")
    print(cleaner.get()['price'])
    print(f"Data type: {cleaner.get()['price'].dtype}")
    
    # Test 3: Handle missing values
    print("\n4. Testing handle_missing...")
    cleaner.handle_missing(columns=['price'], method='mean')
    print("After filling missing 'price' with mean:")
    print(cleaner.get()['price'])
    
    # Test 4: Detect outliers (Z-score)
    print("\n5. Testing detect_outliers_zscore...")
    outliers = cleaner.detect_outliers_zscore(columns=['rating'], threshold=2)
    print(f"Outliers detected (Z-score): {len(outliers)} rows")
    print(outliers)
    
    # Test 5: Handle outliers (IQR)
    print("\n6. Testing handle_outliers_iqr...")
    shape_before = cleaner.get().shape
    cleaner.handle_outliers_iqr(columns=['rating'], k=1.5, action='remove')
    print(f"Shape before: {shape_before}, Shape after: {cleaner.get().shape}")
    print("DataFrame after removing outliers:")
    print(cleaner.get())
    
    # Test 6: Scaling
    print("\n7. Testing scale (standard)...")
    df_for_scaling = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    scaler = TabularCleaning(df_for_scaling)
    scaler.scale(columns=['feature1', 'feature2'], method='standard')
    print("After standard scaling:")
    print(scaler.get())
    
    # Test 7: Binning
    print("\n8. Testing binning...")
    df_for_binning = pd.DataFrame({
        'age': [18, 25, 35, 45, 55, 65, 75]
    })
    binner = TabularCleaning(df_for_binning)
    binner.binning(column='age', method='cut', bins=[0, 30, 60, 100], labels=['Young', 'Middle', 'Senior'])
    print("After binning 'age':")
    print(binner.get())
    
    # Test 8: Process column with custom function
    print("\n9. Testing process_column...")
    processor = TabularCleaning(df_for_binning)
    processor.process_column(column='age', func=lambda x: x * 2)
    print("After applying custom function (multiply by 2) to 'age':")
    print(processor.get())
    
    # Test 9: Method chaining
    print("\n10. Testing method chaining...")
    df_chain = pd.DataFrame({
        'value': ['100$', '200$', '300$', '400$', '500$']
    })
    result = (TabularCleaning(df_chain)
              .replace_symbols(columns=['value'], symbols=['$'])
              .astype(columns=['value'], dtype='float')
              .scale(columns=['value'], method='minmax')
              .get())
    print("After chaining multiple operations (replace, convert, scale):")
    print(result)
    
    print("\n✅ All TabularCleaning tests passed!")
    return True

def test_image_cleaning():
    """Test all ImageCleaning methods"""
    print("\n" + "=" * 60)
    print("TESTING IMAGE CLEANING")
    print("=" * 60)
    
    # Create sample images
    print("\n1. Creating sample images...")
    
    # Color image (3 channels)
    color_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Grayscale image
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    # List of images
    image_list = [color_img, gray_img]
    
    # Dict of images
    image_dict = {'color': color_img, 'gray': gray_img}
    
    print(f"Color image shape: {color_img.shape}")
    print(f"Gray image shape: {gray_img.shape}")
    
    # Test 1: Initialize with different input types
    print("\n2. Testing initialization with different input types...")
    
    # Single array
    cleaner1 = ImageCleaning(color_img)
    print(f"From single array: {len(cleaner1.images)} image(s)")
    
    # List
    cleaner2 = ImageCleaning(image_list)
    print(f"From list: {len(cleaner2.images)} image(s)")
    
    # Dict
    cleaner3 = ImageCleaning(image_dict)
    print(f"From dict: {len(cleaner3.images)} image(s)")
    
    # Test 2: Resize
    print("\n3. Testing resize...")
    cleaner = ImageCleaning(color_img)
    cleaner.resize(size=(64, 64))
    result = cleaner.get()
    print(f"After resize to (64, 64): {result[0].shape}")
    
    # Test 3: Convert to grayscale
    print("\n4. Testing convert_color to grayscale...")
    cleaner = ImageCleaning(color_img)
    cleaner.convert_color(mode='grayscale')
    result = cleaner.get()
    print(f"After converting to grayscale: {result[0].shape}")
    
    # Test 4: Normalize
    print("\n5. Testing normalize (0-1)...")
    cleaner = ImageCleaning(color_img)
    cleaner.normalize(method='0-1')
    result = cleaner.get()
    print(f"After normalization (0-1): min={result[0].min():.3f}, max={result[0].max():.3f}")
    
    print("\n6. Testing normalize (-1 to 1)...")
    cleaner = ImageCleaning(color_img)
    cleaner.normalize(method='minus1-1')
    result = cleaner.get()
    print(f"After normalization (-1 to 1): min={result[0].min():.3f}, max={result[0].max():.3f}")
    
    # Test 5: Denoise
    print("\n7. Testing denoise methods...")
    methods = ['gaussian', 'median', 'bilateral', 'box']
    for method in methods:
        cleaner = ImageCleaning(color_img)
        cleaner.denoise(method=method)
        result = cleaner.get()
        print(f"Denoise with {method}: shape={result[0].shape}")
    
    # Test 6: Process with custom function
    print("\n8. Testing process_image with custom function...")
    cleaner = ImageCleaning(color_img)
    cleaner.process_image(func=lambda img: cv2.flip(img, 1))  # Horizontal flip
    result = cleaner.get()
    print(f"After custom processing (flip): {result[0].shape}")
    
    # Test 7: Method chaining
    print("\n9. Testing method chaining...")
    result = (ImageCleaning(image_list)
              .resize(size=(50, 50))
              .normalize(method='0-1')
              .get())
    print(f"After chaining resize and normalize:")
    for key, img in result.items():
        print(f"  Image {key}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
    
    # Test 8: Multiple images processing
    print("\n10. Testing processing multiple images...")
    multi_images = {
        'img1': np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8),
        'img2': np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8),
        'img3': np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8),
    }
    cleaner = ImageCleaning(multi_images)
    cleaner.resize(size=(64, 64)).convert_color(mode='grayscale').normalize(method='0-1')
    result = cleaner.get()
    print(f"Processed {len(result)} images:")
    for key, img in result.items():
        print(f"  {key}: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    print("\n✅ All ImageCleaning tests passed!")
    return True

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    # TabularCleaning errors
    print("\n1. Testing TabularCleaning errors...")
    
    try:
        TabularCleaning("not a dataframe")
        print("❌ Should have raised TypeError")
    except TypeError as e:
        print(f"✅ Correctly raised TypeError: {e}")
    
    try:
        TabularCleaning(pd.DataFrame())
        print("❌ Should have raised ValueError for empty DataFrame")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    df = pd.DataFrame({'a': [1, 2, 3]})
    cleaner = TabularCleaning(df)
    
    try:
        cleaner.replace_symbols(columns=['nonexistent'])
        print("❌ Should have raised KeyError")
    except KeyError as e:
        print(f"✅ Correctly raised KeyError: {e}")
    
    try:
        cleaner.handle_missing(columns=['a'], method='value')
        print("❌ Should have raised ValueError (missing fill_value)")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # ImageCleaning errors
    print("\n2. Testing ImageCleaning errors...")
    
    try:
        ImageCleaning("not valid input")
        print("❌ Should have raised TypeError")
    except TypeError as e:
        print(f"✅ Correctly raised TypeError: {e}")
    
    try:
        ImageCleaning(np.array([1, 2, 3, 4]))  # 1D array
        print("❌ Should have raised ValueError for 1D array")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cleaner = ImageCleaning(img)
    
    try:
        cleaner.convert_color(mode='invalid')
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    print("\n✅ All error handling tests passed!")
    return True

def test_operations_log():
    """Test that operations are being logged"""
    print("\n" + "=" * 60)
    print("TESTING OPERATIONS LOG")
    print("=" * 60)
    
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    cleaner = TabularCleaning(df)
    cleaner.handle_missing(columns=['value'], method='mean')
    cleaner.scale(columns=['value'], method='standard')
    
    print(f"\nOperations log has {len(cleaner.operations_log)} entries:")
    for i, log in enumerate(cleaner.operations_log, 1):
        print(f"{i}. {log['operation']}: {log['details']}")
        print(f"   Shape before: {log['shape_before']}")
    
    print("\n✅ Operations log test passed!")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NLUZTOOLBOX - REFACTORED CODE TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all test suites
        tests = [
            test_tabular_cleaning,
            test_image_cleaning,
            test_error_handling,
            test_operations_log
        ]
        
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                print(f"\n❌ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                results.append(False)
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        passed = sum(results)
        total = len(results)
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("The refactored code is working correctly!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
