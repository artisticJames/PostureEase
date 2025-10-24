#!/usr/bin/env python3
"""
Camera Detection Test
Test which cameras are available and help you select the right one.
"""

import cv2
import time

def test_cameras():
    """Test all available cameras and show their indices."""
    print("🔍 Testing Available Cameras")
    print("=" * 40)
    
    available_cameras = []
    
    for i in range(5):  # Check first 5 camera indices
        print(f"Testing camera index {i}...", end=" ")
        
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ WORKING ({width}x{height})")
                available_cameras.append(i)
                
                # Show a preview for 3 seconds
                print(f"  Showing preview for camera {i} (press 'q' to skip)...")
                start_time = time.time()
                while time.time() - start_time < 3:
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow(f'Camera {i}', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                cv2.destroyAllWindows()
            else:
                print("❌ No frame")
            cap.release()
        else:
            print("❌ Cannot open")
    
    print(f"\n📊 Summary:")
    print(f"Available cameras: {available_cameras}")
    
    if available_cameras:
        print(f"\n🎯 Recommendations:")
        print(f"  - Camera {available_cameras[0]} will be used by default (highest index)")
        print(f"  - If you want to use a specific camera, set CAMERA_INDEX in config")
        print(f"  - Example: CAMERA_INDEX=1 to force camera index 1")
    else:
        print("❌ No cameras detected!")
    
    return available_cameras

def main():
    """Main test function."""
    print("PosturEase Camera Detection Test")
    print("=" * 50)
    print("This will test all available cameras and show you which one to use.")
    print("Make sure your external camera is connected!")
    print()
    
    input("Press Enter to start camera detection...")
    
    cameras = test_cameras()
    
    print(f"\n🎉 Camera detection complete!")
    print(f"Found {len(cameras)} working camera(s)")
    
    if len(cameras) > 1:
        print(f"\n💡 To use a specific camera, add this to your .env file:")
        print(f"   CAMERA_INDEX=1  # or 2, 3, etc.")
        print(f"   # Leave empty for auto-detection (recommended)")

if __name__ == "__main__":
    main()
