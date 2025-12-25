# FOUNDED BY OPTUNA DEVELOPMENT TEAM & ITMO

import optuna

class TuningStudy:
    def __init__(self, direction="maximize", **study_kwargs):
        self.study = optuna.create_study(
            direction=direction,
            **study_kwargs
        )

    def optimize(self, objective, **optimize_kwargs):
        return self.study.optimize(objective, **optimize_kwargs)

    @property
    def best_params(self):
        return self.study.best_params

    @property
    def best_value(self):
        return self.study.best_value
    
    @property
    def best_trial(self):
        return self.study.best_trial
