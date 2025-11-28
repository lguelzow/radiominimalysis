import iminuit
import numpy as np


class MyMinuitMinimizer(iminuit.Minuit):

    def __init__(self, objec, lmfit_params, objec_kwargs, verbose=False):

        self.objec = objec
        self.lmfit_params = lmfit_params
        self.objec_kwargs = objec_kwargs
        self.verbose = verbose

        name_value_pair, fixed, limits = self.get_minuit_parameter_from_lmfit_init()
        super(MyMinuitMinimizer, self).__init__(self.cost, **
                                                name_value_pair, name=list(name_value_pair.keys()))
        self.errordef = iminuit.Minuit.LEAST_SQUARES
        self.fixed = fixed
        self.limits = limits

        if self.verbose:
            print("Start values:")
            print(self.params)
            print("      Parametrized values")
            for key, value in self.get_minuit_parameter_from_lmfit_expr().items():
                print("    | {:<23} | {:<10} ".format(key, value))
            print('\n')

    def print_result(self, verbose=0):
        if self.verbose or verbose:
            print("End values:")
            print(self.params)
            print("      Parametrized values")
            for key, value in self.get_minuit_parameter_from_lmfit_expr().items():
                print("    | {:<23} | {:<10} ".format(key, value))

    def get_minuit_parameter_from_lmfit_init(self):

        name_value_pair = {}
        vary = []
        limits = []
        for name, param in self.lmfit_params.items():
            if param.expr is None:
                name_value_pair[name] = param.value
                vary.append(param.vary)
                limits.append([param.min, param.max])

        fixed = ~np.array(vary)
        return name_value_pair, fixed, limits

    def get_minuit_parameter_from_lmfit_expr(self, dmax=None):
        if dmax is not None:
            self.lmfit_params["distance_xmax_geometric"].set(value=dmax)

        name_value_pair = {}
        for name, param in self.lmfit_params.items():
            if param.expr is not None:
                name_value_pair[name] = self.lmfit_params.eval(param.expr)

        return name_value_pair

    def cost(self, *pars):

        dmax = pars[self.parameters.index("distance_xmax_geometric")]

        params = self.get_minuit_parameter_from_lmfit_expr(dmax)
        for name, value in zip(self.parameters, pars):
            if name in params:
                continue  # dont overwrite param. parameters

            params[name] = value

        r = self.objec(
            params, **self.objec_kwargs)

        return r

    def get_all_param_value_dict(self):

        params = self.get_minuit_parameter_from_lmfit_expr()
        for name, value in zip(self.parameters, self.values):
            if name in params:
                continue  # dont overwrite param. parameters

            params[name] = value

        return params

    def get_all_param_error_dict(self):

        error = self.get_minuit_parameter_from_lmfit_expr()
        error = {key: 0 for key in error}
        for name, value in zip(self.parameters, self.errors):
            if name in error:
                continue  # dont overwrite param. parameters

            error[name] = value

        return error
