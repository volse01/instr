import cv2



def main():# Example usage:
    input_image_path = './../data/ipa/synthetic/1/1_class_segmaps.png'
    output_image = resize_keep_centered_greyscale(input_image_path, target_width=640, target_height=480)

    # Display the output image
    cv2.imshow('Optimized Resized Image with Centered Crop', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
