# Legacy PyBullet implementation

This is the original simulator, kept for reference only. **Nothing in the
project imports it** -- the live code is in `../jenga_sim/`.

It is here because the port to MuJoCo was driven by measurements worth
keeping, and this is what they were measured against. PyBullet's contact
tolerances are absolute distances of a few millimetres, which is a large
fraction of a real 15 mm Jenga block, so this version had to simulate a tower
10x oversized to stay stable. See "Why MuJoCo, and why real scale" in the main
README for the head-to-head numbers.

These files will not run as-is: they expect `pybullet`, which is no longer in
`requirements.txt`, and the patched macOS wheel they needed is not committed.
