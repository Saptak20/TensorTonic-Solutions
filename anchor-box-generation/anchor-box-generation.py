def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """

    anchors = []

    # Compute stride
    stride = image_size / feature_size

    # Iterate over grid cells
    for i in range(feature_size):
        for j in range(feature_size):

            # Center coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # Generate anchors for each scale and aspect ratio
            for s in scales:
                for r in aspect_ratios:

                    w = s * (r ** 0.5)
                    h = s / (r ** 0.5)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors