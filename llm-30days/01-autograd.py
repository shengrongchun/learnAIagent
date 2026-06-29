class Value:
  def __init__(self, data, _children=()): # __init__ 是给对象初始化，不需要有返回值
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._backward = lambda: None
  
  def __add__(self, other): # __add__是一个实例的方法，如果需要返回值就必须主动返回
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other))
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other))
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    # 我需要根据当前实例倒退之前的若干计算实例 比如 l = a+b 那么当前实例是l,之前的计算实例是a和b
    topo = []
    visited = set()
    def build(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build(child)
        topo.append(v)
    build(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

# 调整参数更新公式 p = p - learning_rate * grad learning_rate=0.01
# a, b, c, f = Value(2), Value(-3.0), Value(10.0), Value(-2.0)
a, b, c, f = Value(2.2), Value(-3.204), Value(9.098), Value(-1.096)
L = (a*b+c)*f
L.backward()
print('L =', L.data)
print('a.grad =', a.grad, 'b.grad =', b.grad, 'c.grad =', c.grad, 'f.grad =', f.grad)
