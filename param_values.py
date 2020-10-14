def set_default_values(args, also_hyper_params=True):
    # -set default-values for certain arguments based on chosen scenario & experiment
    args.tasks = (
        (5 if args.experiment == "splitMNIST" else 10) if args.tasks is None else args.tasks
    )
    args.iters = (
        (2000 if args.experiment == "splitMNIST" else 5000) if args.iters is None else args.iters
    )
    args.lr = (0.001 if args.experiment == "splitMNIST" else 0.0001) if args.lr is None else args.lr
    args.fc_units = (
        (400 if args.experiment == "splitMNIST" else 1000)
        if args.fc_units is None
        else args.fc_units
    )

    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
    # -if [iCaRL] is selected, select all accompanying options
    if hasattr(args, "icarl") and args.icarl:
        args.use_exemplars = True
        args.add_exemplars = True
        args.bce = True
        args.bce_distill = True

    if also_hyper_params:
        if args.scenario == "task":
            args.gating_prop = (
                (0.95 if args.experiment == "splitMNIST" else 0.55)
                if args.gating_prop is None
                else args.gating_prop
            )
            args.si_c = (
                (50.0 if args.experiment == "splitMNIST" else 5.0)
                if args.si_c is None
                else args.si_c
            )
            args.ewc_lambda = (
                (10000000.0 if args.experiment == "splitMNIST" else 500.0)
                if args.ewc_lambda is None
                else args.ewc_lambda
            )
            if hasattr(args, "o_lambda"):
                args.o_lambda = (
                    (100000000.0 if args.experiment == "splitMNIST" else 500.0)
                    if args.o_lambda is None
                    else args.o_lambda
                )
            args.gamma = (
                (0.8 if args.experiment == "splitMNIST" else 0.8)
                if args.gamma is None
                else args.gamma
            )
        elif args.scenario == "domain":
            args.si_c = (
                (500.0 if args.experiment == "splitMNIST" else 5.0)
                if args.si_c is None
                else args.si_c
            )
            args.ewc_lambda = (
                (1000000.0 if args.experiment == "splitMNIST" else 500.0)
                if args.ewc_lambda is None
                else args.ewc_lambda
            )
            if hasattr(args, "o_lambda"):
                args.o_lambda = (
                    (100000000.0 if args.experiment == "splitMNIST" else 1000.0)
                    if args.o_lambda is None
                    else args.o_lambda
                )
            args.gamma = (
                (0.7 if args.experiment == "splitMNIST" else 0.9)
                if args.gamma is None
                else args.gamma
            )
        elif args.scenario == "class":
            args.si_c = (
                (0.5 if args.experiment == "splitMNIST" else 0.1)
                if args.si_c is None
                else args.si_c
            )
            args.ewc_lambda = (
                (100000000.0 if args.experiment == "splitMNIST" else 1.0)
                if args.ewc_lambda is None
                else args.ewc_lambda
            )
            if hasattr(args, "o_lambda"):
                args.o_lambda = (
                    (1000000000.0 if args.experiment == "splitMNIST" else 5.0)
                    if args.o_lambda is None
                    else args.o_lambda
                )
            args.gamma = (
                (0.8 if args.experiment == "splitMNIST" else 1.0)
                if args.gamma is None
                else args.gamma
            )
    return args


def validate_args(args):
    # -if XdG is selected but not the Task-IL scenario, give error
    if (not args.scenario == "task") and args.xdg:
        raise ValueError("'XdG' is only compatible with the Task-IL scenario.")
    # -if EWC, SI, XdG, A-GEM or iCaRL is selected together with 'feedback', give error
    if args.feedback and (args.ewc or args.si or args.xdg or args.icarl or args.agem):
        raise NotImplementedError(
            "EWC, SI, XdG, A-GEM and iCaRL are not supported with feedback connections."
        )
    # -if A-GEM is selected without any replay, give warning
    if args.agem and args.replay == "none":
        raise Warning(
            "The '--agem' flag is selected, but without any type of replay. "
            "For the original A-GEM method, also select --replay='exemplars'."
        )
    # -if EWC, SI, XdG, A-GEM or iCaRL is selected together with offline-replay, give error
    if args.replay == "offline" and (args.ewc or args.si or args.xdg or args.icarl or args.agem):
        raise NotImplementedError(
            "Offline replay cannot be combined with EWC, SI, XdG, A-GEM or iCaRL."
        )
    # -if binary classification loss is selected together with 'feedback', give error
    if args.feedback and args.bce:
        raise NotImplementedError(
            "Binary classification loss not supported with feedback connections."
        )
    # -if XdG is selected together with both replay and EWC, give error (either one of them alone with XdG is fine)
    if (
        (args.xdg and args.gating_prop > 0)
        and (not args.replay == "none")
        and (args.ewc or args.si)
    ):
        raise NotImplementedError(
            "XdG is not supported with both '{}' replay and EWC / SI.".format(args.replay)
        )
        # --> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating backward step on EWC/SI-loss also for each mask separately)
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario == "class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
