import cv2
import json
import os
import glob

def annotate_panoramas(image_folder, output_json="ground_truth.json"):
    """
    Opens panoramas in a folder one by one. 
    Click on street entrances to record their yaw angles.
    Press 'q' or 'Enter' to save and move to the next image.
    """
    # Find all JPGs/PNGs in the folder
    image_paths = glob.glob(os.path.join(image_folder, "*.[jp][pn]g"))
    dataset = []

    print(f"Found {len(image_paths)} images to annotate.")
    print("INSTRUCTIONS:")
    print(" - Left Click: Mark a street entrance")
    print(" - 'c': Clear all marks on the current image")
    print(" - 'q' or 'Enter': Save current image and move to the next")
    print("-" * 30)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        H, W = img.shape[:2]
        
        # Resize for display if the image is massive (e.g., 4K+)
        display_scale = 1.0
        max_width = 1600
        if W > max_width:
            display_scale = max_width / W
            img = cv2.resize(img, (max_width, int(H * display_scale)))
            H, W = img.shape[:2]

        display_img = img.copy()
        current_angles = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Calculate the yaw angle (-180 to 180) based on X coordinate
                yaw = (x / W) * 360.0 - 180.0
                current_angles.append(yaw)
                
                # Draw a vertical line and a dot to confirm the click
                cv2.line(display_img, (x, 0), (x, H), (0, 255, 0), 2)
                cv2.circle(display_img, (x, y), 6, (0, 0, 255), -1)
                
                # Add text label
                cv2.putText(display_img, f"{yaw:.1f}", (x + 10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Panorama Annotator", display_img)
                print(f"  -> Marked angle: {yaw:.1f}°")

        cv2.namedWindow("Panorama Annotator")
        cv2.setMouseCallback("Panorama Annotator", click_event)
        
        while True:
            cv2.imshow("Panorama Annotator", display_img)
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'q' or 'Enter' to finish this image
            if key == ord('q') or key == 13: 
                break
            # Press 'c' to clear/reset the current image
            elif key == ord('c'):
                print("  -> Cleared current angles.")
                current_angles.clear()
                display_img = img.copy()
                
        cv2.destroyAllWindows()
        
        # Save to our dataset array
        dataset.append({
            "image_path": img_path,
            "true_yaws": sorted(current_angles)
        })
        print(f"Saved {len(current_angles)} angles for {os.path.basename(img_path)}\n")

    # Save the final dataset to a JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"✅ Annotation complete! Saved to {output_json}")
    return dataset

if __name__ == "__main__":
    # Point this to a folder full of test panoramas to annote them
    annotate_panoramas("./streetview_panoramas")