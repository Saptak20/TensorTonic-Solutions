def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    H = len(image)
    W = len(image[0])
    
    y_scale = (H - 1) / (new_h - 1) if new_h > 1 else 0.0
    x_scale = (W - 1) / (new_w - 1) if new_w > 1 else 0.0
    
    out = []
    for i in range(new_h):
        y = i * y_scale
        y0 = int(y)
        y1 = min(y0 + 1, H - 1)
        dy = y - y0
        
        row = []
        for j in range(new_w):
            x = j * x_scale
            x0 = int(x)
            x1 = min(x0 + 1, W - 1)
            dx = x - x0
            
            v0 = image[y0][x0] * (1 - dx) + image[y0][x1] * dx
            v1 = image[y1][x0] * (1 - dx) + image[y1][x1] * dx
            
            row.append(v0 * (1 - dy) + v1 * dy)
        out.append(row)
        
    return out