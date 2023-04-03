import numpy as np


def gsf(reverb_speech, params):

    win = np.sqrt(np.hanning(params.wlen))
    frames = (len(reverb_speech) - params.wlen) // params.inc + 1
    half = params.wlen // 2 + 1
    ''' (1) Normalized Reverberant Speech Spectrogram Generation '''
    x_final = np.zeros((frames, half))
    x_angle = np.zeros((frames, half))
    k = 0
    for i in range(frames):
        x_fft = np.fft.rfft(win * reverb_speech[k:k + params.wlen])
        x_angle[i, :] = np.angle(x_fft)
        x_final[i, :] = 20 * np.log10(np.abs(x_fft) ** 2)
        k += params.inc
    x_hat = np.where(x_final < (np.max(x_final) - 255), np.max(x_final) - 255, x_final)  # (7)
    x_bar = x_hat - np.min(x_hat) + 1e-05  # (6)
    guidance_spectrogram = x_bar / np.max(x_bar)  # (5)
    ''' (2) Guided Filter '''
    N = (2 * params.r1 + 1) * (2 * params.r2 + 1)  # the number of time-frequency bins
    input_spectrogram = np.zeros((frames, half))
    input_spectrogram[0, :] = guidance_spectrogram[0, :]
    for i in range(1, frames):
        input_spectrogram[i, :] = params.beta * input_spectrogram[i - 1, :] + (1 - params.beta) * guidance_spectrogram[i, :]  # (8)
    m_gs = np.zeros((frames, half))
    m_gs_2 = np.zeros((frames, half))
    m_is = np.zeros((frames, half))
    m_gs_is = np.zeros((frames, half))
    A = np.zeros((frames, half))
    B = np.zeros((frames, half))
    for i in range(frames):
        frame_min = max(0, i - params.r2)
        frame_max = min(frames, i + params.r2)
        for j in range(half):
            freq_min = max(0, j - params.r1)
            freq_max = min(half, j + params.r1)
            local_guidance_spectrogram = guidance_spectrogram[frame_min:frame_max, freq_min:freq_max]
            local_input_spectrogram = input_spectrogram[frame_min:frame_max, freq_min:freq_max]
            m_gs_2[i][j] = (local_guidance_spectrogram ** 2).sum() / N  # (11)
            m_gs[i][j] = local_guidance_spectrogram.sum() / N  # (12)
            m_gs_is[i][j] = (local_guidance_spectrogram * local_input_spectrogram).sum() / N  # (14)
            m_is[i][j] = local_input_spectrogram.sum() / N  # (15)
    sigma_gs_2 = m_gs_2 - m_gs ** 2  # (10)
    cov_gs_is = m_gs_is - m_is * m_gs  # (13)
    A_tilde = cov_gs_is / (sigma_gs_2 + params.epsilon)  # (16)
    B_tilde = m_is - A_tilde * m_gs  # (17)
    for i in range(frames):
        frame_min = max(0, i - params.r2)
        frame_max = min(frames, i + params.r2)
        for j in range(half):
            freq_min = max(0, j - params.r1)
            freq_max = min(half, j + params.r1)
            local_A_tilde = A_tilde[frame_min:frame_max, freq_min:freq_max]
            local_B_tilde = B_tilde[frame_min:frame_max, freq_min:freq_max]
            A[i][j] = local_A_tilde.sum() / N  # (18)
            B[i][j] = local_B_tilde.sum() / N  # (19)
    output_spectrogram = A * guidance_spectrogram + B  # (9)
    ''' (3) Speech Reconstruction '''
    y = np.where(guidance_spectrogram - params.alpha * output_spectrogram < 0, 0, guidance_spectrogram - params.alpha * output_spectrogram)  # (20)
    y_tilde = y / np.max(y)  # (21)
    gain_1 = np.where(y_tilde / guidance_spectrogram < params.gain_min, params.gain_min, y_tilde / guidance_spectrogram)
    gain = np.where(gain_1 > params.gain_max, params.gain_max, gain_1)  # (22)
    out = gain * x_final
    output = np.zeros((frames * params.inc + params.overlap))
    k = 0
    overlap_buff = np.zeros(params.wlen)
    for i in range(frames):
        buff = np.sqrt(np.power(10, out[i, :] / 20)) * np.exp(1j * x_angle[i, :])
        segment = np.fft.irfft(buff) * win
        output[k:k + params.overlap] = overlap_buff[params.inc:] + segment[:params.overlap]
        overlap_buff = segment
        k += params.inc
    return output