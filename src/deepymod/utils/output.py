import sys, time


def progress(iteration, start_time, max_iteration, cost, MSE, PI, L1):
    '''Prints and updates progress of training cycle in command line.'''
    percent = iteration.item()/max_iteration * 100
    elapsed_time = time.time() - start_time
    time_left = elapsed_time * (max_iteration/iteration - 1) if iteration.item() != 0 else 0
    sys.stdout.write(f"\r  {iteration:>9}   {percent:>7.2f}%   {time_left:>13.0f}s   {cost:>8.2e}   {MSE:>8.2e}   {PI:>8.2e}   {L1:>8.2e} ")
    sys.stdout.flush()