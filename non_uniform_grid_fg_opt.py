def emit_all(...):
    # Previous code...
    if np.isscalar(v_scalar):
        V_list.append(v_scalar)
    else:
        filtered_v_scalar = v_scalar[mask]
        V_list.extend(filtered_v_scalar)
    # Continue with the rest of your function...