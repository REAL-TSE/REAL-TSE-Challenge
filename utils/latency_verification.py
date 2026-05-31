import torch


def estimate_latency(
    model,
    fs=16000,
    signal_len=8,
    enroll_len=5,
    perturb_starts=(1.000, 2.000, 1.234, 2.346, 3.459), # Include both integer-second and non-aligned perturbation.
    threshold=1e-5,
):
    """
    Reference script for causality / latency consistency verification.
    The verification script assumes deterministic inference. Any stochastic
    sampling or persistent internal states should be disabled or properly
    reset during verification.

    This script is intended to verify whether a streaming system behaves
    consistently with its claimed algorithmic latency. The reported value
    should be interpreted as a future-dependency estimate rather than an
    exact latency measurement.

    Notes:
        - The measured value may vary depending on perturbation position,
          model architecture, numerical precision, and detection threshold.
        - The result is intended for verification purposes only and should
          not be interpreted as the official algorithmic latency.
        - Participants should report the nominal/theoretical latency of
          their system separately.
        - Any look-ahead, chunk accumulation, output buffering, or other
          latency-inducing operations must be included in the reported
          algorithmic latency for the Online Track.
        - The default perturbation settings and threshold should be used
          when reporting results.
        - For frame/chunk-based systems, the measured value may differ
          from the reported latency by up to approximately one hop/chunk
          duration depending on perturbation alignment. Such deviations
          are expected and acceptable.
        - Additional output buffering may partially or completely offset
          the measured future dependency. Therefore, participants should
          provide a detailed explanation if the measured result differs
          significantly from the reported latency.
    """

    model.eval()

    with torch.no_grad():

        # Random test input and enrollment signal.
        x = torch.randn(1, fs * signal_len).clamp_(-1, 1)
        enroll = torch.randn(1, fs * enroll_len).clamp_(-1, 1)

        # For the official baseline (WESep), the model returns a tuple/list
        # and the enhanced waveform is the first element.
        # Participants may need to modify this line according to their own
        # model implementation.
        y1 = model(x, enroll)[0]

        results = []

        for perturb_start in perturb_starts:

            x2 = x.clone()

            start = int(perturb_start * fs)

            # Apply a perturbation to all future samples.
            #
            # The perturbation magnitude and form are intentionally fixed
            # to provide a consistent verification procedure across
            # participants.
            #
            # Participants are encouraged to use the default perturbation
            # when reporting results. Alternative perturbations may be
            # explored for analysis purposes, but reported values should
            # be based on the default configuration below.
            x2[..., start:] += 0.5 * torch.randn_like(x2[..., start:])

            # See note above regarding model outputs.
            y2 = model(x2, enroll)[0]

            diff = (y1 - y2).abs()

            # Detection threshold.
            #
            # A value of 1e-5 is recommended as a compromise between
            # numerical robustness and sensitivity. Smaller values may
            # become sensitive to floating-point noise and hardware
            # differences.
            idx = (diff > threshold).float().argmax()

            response_time = idx.item() / fs

            # Future dependency estimate.
            #
            # Positive values indicate that future input samples influence
            # the current output (e.g., look-ahead, frame-based processing,
            # chunk-based processing).
            #
            # The measured value should not be interpreted as an exact
            # latency measurement. In particular, additional output
            # buffering may reduce or even offset the measured future
            # dependency.
            #
            # Therefore, participants should report the complete
            # algorithmic latency of their systems, including all
            # look-ahead, chunk accumulation, and output buffering
            # operations.
            #
            # For frame/chunk-based systems, moderate variations caused by
            # perturbation alignment within a frame/chunk are expected.
            future_dependency = perturb_start - response_time

            print(
                f"Perturbation @ {perturb_start:.3f}s | "
                f"Response @ {response_time:.3f}s | "
                f"Future dependency = "
                f"{future_dependency * 1000:.1f} ms"
            )

            results.append(future_dependency)

        print(
            "\nSummary:\n"
            f"Min  : {min(results) * 1000:.1f} ms\n"
            f"Mean : {sum(results) / len(results) * 1000:.1f} ms\n"
            f"Max  : {max(results) * 1000:.1f} ms"
        )

        print(
            "\nNote: The values above are intended for causality and "
            "latency-consistency verification only. Participants must "
            "separately report the total algorithmic latency of their "
            "systems, including any look-ahead, chunk accumulation, "
            "and output buffering."
        )

        return results