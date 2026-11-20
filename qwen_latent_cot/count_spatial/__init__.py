"""Count and spatial edit pairs built backwards from verified T2I images.

The correct image is the training target. An editor makes it wrong, a VLM
observes both images, and the written instruction reports the difference it
actually saw, never the instruction that was planned.
"""
