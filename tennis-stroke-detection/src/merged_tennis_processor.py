def process_single_video(video_path, llc_path=None, video_id=None):
    """
    Process a single video with enhanced pose detection, visualization, and segmentation
    
    Args:
        video_path: Path to the video file
        llc_path: Path to the LLC file
        video_id: Video ID to use
        
    Returns:
        bool: Success or failure
    """
    try:
        # Get video ID if not provided
        if video_id is None:
            video_id = get_next_video_id()
        
        # Get output prefix
        output_prefix = f"video_{video_id}"
        
        print(f"\n=== PROCESSING VIDEO AS {output_prefix} ===")
        
        # Step 1: Process video with MediaPipe to create enhanced visualizations
        print(f"[STEP 1] Processing video with MediaPipe")
        skeleton_output, overlay_output, landmarks_dict_list, csv_output = process_video_with_mediapipe(
            video_path, output_prefix)
        
        if not csv_output or not os.path.exists(csv_output):
            print(f"[ERROR] Failed to process video or create CSV output")
            return False
        
        # Step 2: Normalize data
        norm_csv = f"{OUTPUT_DIR}/{output_prefix}_normalized.csv"
        normalize_pose_data(csv_output, norm_csv)
        
        # If LLC path is provided, setup for pipeline
        if llc_path and os.path.exists(llc_path):
            # Step 3: Set up files for pipeline
            llc_processed = setup_for_pipeline(
                video_path, skeleton_output, overlay_output, 
                csv_output, norm_csv, llc_path, video_id
            )
            
            # Step 4: Run stroke segmentation if LLC file has valid annotations
            if check_llc_file(llc_path):
                print(f"[INFO] Valid LLC file found, running stroke segmentation")
                # Get the video path in the videos directory
                processed_video_path = os.path.join(VIDEOS_DIR, f"video_{video_id}", f"video_{video_id}.mp4")
                if os.path.exists(processed_video_path):
                    # Run segmentation on the processed video file
                    run_stroke_segmentation(processed_video_path)
                    
                    # Step 5: Ensure strokes are copied to the library
                    created_strokes = manually_copy_clips_to_library(video_id)
                    if created_strokes:
                        print(f"[INFO] Successfully created {len(created_strokes)} strokes in the library.")
                    else:
                        print(f"[WARN] No strokes were created in the library. Please check the segmentation output.")
                else:
                    print(f"[ERROR] Processed video not found at {processed_video_path}")
            else:
                print(f"[WARN] LLC file has no valid stroke annotations, skipping segmentation")
        else:
            print(f"[WARN] No LLC file provided, skipping pipeline setup and segmentation")
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Processed {os.path.basename(video_path)} as {output_prefix}")
        print(f"Results available in Strokes_Library directory")
        
        return True
    except Exception as e:
        import traceback
        print(f"[ERROR] Processing video: {e}")
        print(traceback.format_exc())
        return False 