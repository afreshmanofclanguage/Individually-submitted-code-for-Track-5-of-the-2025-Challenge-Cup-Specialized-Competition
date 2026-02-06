import sys
from submission.task_solver.task_solver import TaskSolver
from tongverse.env import Env, app

TASK_SEED = None

def main(task_id=1):
    assert task_id in [1, 2], "task_id must be either 1 or 2"
    task_id = "TaskOne" if task_id == 1 else "TaskTwo"

    import pdb; pdb.set_trace()
    
    env = Env(seed=TASK_SEED, task_id=task_id) 
    env.reset()

    task_params = env.get_task_params()
    print("TASK INFO")
    print(task_params)

    agent_params = env.get_robot_params()
    print("AGENT INFO")
    print(f"{agent_params}")

    action = {}

    # pylint: disable=undefined-variable
    solver = TaskSolver(task_params, agent_params)  # noqa: F821

    while app.is_running():
        obs, is_done = env.step(action)

        # If the simulation is stopped or task is finished, then exit simulation.
        # Important: Please do not remove this line.
        if is_done:
            print(obs.get("extras"))
            break

        action = solver.next_action(obs)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        task_id = 1
    else:
        task_id = int(sys.argv[1])
        
    main(task_id)
    app.close()
