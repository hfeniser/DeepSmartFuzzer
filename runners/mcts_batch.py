

mcts = RLforDL_MCTS(test_input.shape, input_lower_limit, input_upper_limit,\
     action_division_p1, actions_p2, tc1, tc2, tc3, with_implicit_reward=args.implicit_reward, verbose_image=True)

for i in range(1, 1000):
    test_input, test_label = input_chooser(batch_size=64)
    root, best_input, best_coverage = mcts.run(test_input, coverage)
    if best_coverage > 0:
        input_chooser.append(best_input, test_label)
        coverage.step(best_input, update_state=True)
        print("IMAGE %g SUCCEED" % (i))
        print("found coverage increase", best_coverage)
        print("found different input", np.any(best_input-test_input != 0))
        print("Current Total Coverage", coverage.get_current_coverage())
    else:
        print("IMAGE %g FAILED" % (i))