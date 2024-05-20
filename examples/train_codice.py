from easy_runner import EasyRunner

if __name__ == "__main__":

    exp_name = "benchmark"
    runner = EasyRunner(log_name=exp_name)

    task = [
        "OfflinePointGoal1Gymnasium-v0",
        "OfflineCarGoal1Gymnasium-v0",
        "OfflinePointPush1Gymnasium-v0",
        "OfflineCarPush1Gymnasium-v0",
        "OfflinePointButton1Gymnasium-v0",
        "OfflineCarButton1Gymnasium-v0",
        "OfflineHopperVelocity-v1",
        "OfflineWalker2dVelocity-v1",
    ]
    policy = ["train_coptidice"]
    alpha = ["0.5"]
    cost_limit = ["40"]
    seed = ["42", "420", "4200"]

    # Do not write & to the end of the command, it will be added automatically.
    template = "nohup python examples/train/{}.py --alpha {} --task {} --cost_limit {} --seed {} --update_steps 300000 --device cpu"

    train_instructions = runner.compose(template, [policy, alpha, task, cost_limit, seed])
    runner.start(train_instructions, max_parallel=15)
