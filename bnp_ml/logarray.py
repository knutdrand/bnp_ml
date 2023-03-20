import numpy as np


class LogArray:
    def __init__(self, log_values=np.ndarray, sign: int = 1):
        self._log_values = log_values
        self._sign = sign



    def __str__(self):
        return f'log_array({self._sign}{np.exp(self._log_values)})'

    def __array_ufunc__(self, ufunc: callable, method: str, *inputs, **kwargs):
        """Handle numpy unfuncs called on the runlength array
        
        Currently only handles '__call__' modes and unary and binary functions

        Parameters
        ----------
        ufunc : callable
        method : str
        *inputs :
        **kwargs :
        """
        if method not in ("__call__"):
            return NotImplemented
        if len(inputs) == 1:
            return self.__class__(self._events, ufunc(self._values))
        assert len(inputs) == 2, f"Only unary and binary operations supported for runlengtharray {len(inputs)}"
        
        if ufunc == np.add:
            if not isinstance(other, LogArray):
                other = LogArray(np.log(
            return self.__class__(
                np.logaddexp(*inputs, **kwargs))
        if ufunc == np.subtract:
            pass

        if isinstance(inputs[1], Number):
            return self.__class__(self._events, ufunc(self._values, inputs[1]))
        elif isinstance(inputs[0], Number):
            return self.__class__(self._events, ufunc(inputs[0], self._values))
        return self._apply_binary_func(*inputs, ufunc)
