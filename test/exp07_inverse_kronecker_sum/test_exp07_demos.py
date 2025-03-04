"""Run demos from exp07."""

from kfac_pinns_exp.exp07_inverse_kronecker_sum import demo, demo_class


def test_demo():
    """Execute demo that verifies inverse-of-Kronecker-sum equation."""
    demo.main()


def test_demo_class():
    """Execute demo that verifies inverse-of-Kronecker-sum class."""
    demo_class.main()
