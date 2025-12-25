# FOUNDED BY OPTUNA DEVELOPMENT TEAM & ITMO

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
class TuningStudy:
    def __init__(self, direction="maximize", **study_kwargs):
        self.study = optuna.create_study(
            direction=direction,
            **study_kwargs
        )

    def optimize(self, objective, n_trials=None, verbose=False, **optimize_kwargs):
        def verbose_objective(trial):
            value = objective(trial)
            if verbose:
                print(f"Trial {trial.number}: Value = {value}, Params = {trial.params}")
            return value
        return self.study.optimize(verbose_objective, n_trials=n_trials, **optimize_kwargs)

    @property
    def best_params(self):
        return self.study.best_params

    @property
    def best_value(self):
        return self.study.best_value
    
    @property
    def best_trial(self):
        return self.study.best_trial
