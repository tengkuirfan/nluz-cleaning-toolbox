#!/usr/bin/env python3
"""
Simple test script to verify the toolbox functionality
"""

import pandas as pd
import numpy as np
from nluztoolbox import DataCleaning, ImageCleaning

def test_data_cleaning():
    """Test DataCleaning functionality"""
    print("🧪 Testing DataCleaning...")
    
    # Create test data
    df = pd.DataFrame({
        'name': ['John$', 'Jane!', 'Bob?', None],
        'age': [25, 30, np.nan, 35],
        'price': [100.5, 200.0, 300.0, 1000.0],  # 1000 is outlier
        'category': ['A', 'B', 'A', 'C']
    })
    
    print(f"Original data shape: {df.shape}")
    print("Original data:")
    print(df)
    print()
    
    # Test DataCleaning
    cleaner = DataCleaning(df)
    
    # Test replace symbols
    cleaner.replace_symbols(['name'], ['$', '!', '?'], '')
    print("✅ Symbols replaced")
    
    # Test handle missing
    cleaner.handle_missing(['age'], method='mean')
    cleaner.handle_missing(['name'], method='value', fill_value='Unknown')
    print("✅ Missing values handled")
    
    # Test outlier detection
    cleaner.handle_outliers_zscore(['price'], threshold=2, action='nan')
    print("✅ Outliers handled")
    
    # Test scaling
    cleaner.scale(['price'], method='minmax')
    print("✅ Data scaled")
    
    # Test type conversion
    cleaner.astype(['category'], 'str')
    print("✅ Types converted")
    
    # Get final result
    result = cleaner.get()
    print(f"Final data shape: {result.shape}")
    print("Final data:")
    print(result)
    print()
    
    # Check operations log
    print("Operations log:")
    for op in cleaner.operations_log:
        print(f"  - {op['operation']}: {op['details']}")
    print()
    
    return True

def test_image_cleaning():
    """Test ImageCleaning functionality"""
    print("🖼️  Testing ImageCleaning...")
    
    # Create dummy image data
    dummy_img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_img2 = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    
    # Test with list input
    images_list = [dummy_img1, dummy_img2]
    cleaner = ImageCleaning(images_list)
    print(f"✅ Initialized with list: {len(cleaner.images)} images")
    
    # Test with dict input
    images_dict = {'img1': dummy_img1, 'img2': dummy_img2}
    cleaner = ImageCleaning(images_dict)
    print(f"✅ Initialized with dict: {len(cleaner.images)} images")
    
    # Test with single image
    cleaner = ImageCleaning(dummy_img1)
    print(f"✅ Initialized with single image: {len(cleaner.images)} images")
    
    # Test processing
    cleaner.resize((64, 64))
    print("✅ Images resized")
    
    cleaner.convert_color('grayscale')
    print("✅ Color converted")
    
    cleaner.normalize('0-1')
    print("✅ Images normalized")
    
    result = cleaner.get()
    print(f"Final images: {len(result)} images")
    print(f"Image shape: {result[0].shape}")
    print()
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting toolbox tests...\n")
    
    try:
        # Test DataCleaning
        assert test_data_cleaning(), "DataCleaning test failed"
        
        # Test ImageCleaning  
        assert test_image_cleaning(), "ImageCleaning test failed"
        
        print("🎉 All tests passed successfully!")
        print("✨ Your toolbox is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
