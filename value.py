"""
Stage 2: Autograd engine, built from scratch - no torch.autograd.

This file is a SCAFFOLD, not a solution. The Value class's public
contract (the methods below) is the whole thing. Fill in the bodies
yourself.

Recap of the mechanism (see Stage 1 theory + the worked traces for the
full derivation):
  - A Value wraps a number but also remembers what produced it (its
    children) and which operation created it, so the graph can be
    walked backward later. Plain eager arithmetic (just floats) throws
    this information away the moment it's computed - that's why Value
    needs to exist at all.
  - __add__ and __mul__ build the graph as a side effect of normal-
    looking arithmetic: each one computes the forward result AND
    records enough information (self, other, and which operation) for
    backward() to later know the LOCAL derivative rule to apply:
        d(a+b)/da = 1,       d(a+b)/db = 1
        d(a*b)/da = b.data,  d(a*b)/db = a.data
  - backward() has two jobs: (1) walk the graph in REVERSE topological
    order (never finalize a node's gradient until everything that
    consumed it has already contributed), and (2) at each node, use
    the chain rule to push gradient into its children - ACCUMULATING
    (+=) rather than overwriting, since a Value can be a child of more
    than one node (see the a*b + a*c worked trace - a's gradient is
    the SUM of both branches' contributions, not just one).
  - self.grad must start at 1 for whichever Value backward() is called
    on (dL/dL = 1 - a value's gradient with respect to itself), and 0
    for every other Value before any contributions arrive.

You will need to decide for yourself: how a Value remembers its
children, how it remembers which local-derivative rule to use when
backward() reaches it, and how you build the reverse-topological
order. All of that is your design - nothing here prescribes it.
"""


class Value:
    def __init__(self, data):
        """
        Wrap a plain number in a Value. .grad starts at 0 - this
        Value hasn't received any gradient yet, and won't, unless
        backward() is eventually called on something downstream of it
        (or on this Value itself).
        """
        self.data = data
        self.grad = 0

    def __add__(self, other):
        """
        Return a NEW Value representing self + other. Must build the
        graph (remember self and other as this new Value's children)
        so backward() can later find its way back to them.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """
        Return a NEW Value representing self * other. Same graph-
        building requirement as __add__.
        """
        raise NotImplementedError

    def backward(self):
        """
        Compute the gradient of every Value that fed into this one
        (directly or indirectly), and store it in each Value's .grad.
        Seeds self.grad = 1 first (this Value's gradient with respect
        to itself). Must accumulate contributions (+=), never
        overwrite, for any Value used in more than one place.
        """
        raise NotImplementedError
