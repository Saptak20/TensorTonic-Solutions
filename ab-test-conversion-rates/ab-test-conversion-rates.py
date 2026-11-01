import math
from statistics import NormalDist

def ab_test_conversion(control_conversions: int, control_n: int, treatment_conversions: int, treatment_n: int, alpha: float = 0.05) -> dict:
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    
    lift = treatment_rate - control_rate
    
    pooled_rate = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    pooled_se = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / control_n + 1 / treatment_n))
    
    if pooled_se == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = lift / pooled_se
        p_value = 2 * (1 - NormalDist().cdf(abs(z)))
        
    unpooled_se = math.sqrt(
        (control_rate * (1 - control_rate) / control_n) + 
        (treatment_rate * (1 - treatment_rate) / treatment_n)
    )
    
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    
    ci_low = lift - (z_alpha * unpooled_se)
    ci_high = lift + (z_alpha * unpooled_se)
    
    significant = p_value < alpha
    
    return {
        "lift": round(lift, 6),
        "z": round(z, 6),
        "p_value": round(p_value, 6),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "significant": significant
    }