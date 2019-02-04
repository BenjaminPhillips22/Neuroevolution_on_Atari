
# Run these and check output

# rs_tests/my_test_c1.py
# 6 seeds, small model, CPU, sequentially
# does the reward for seed 5 equal 160? (others will be zero)
# do the frame counts make sense? 
# (on my computer seed 0 runs for 20000 frames which I think is an error due
# to not moving once, though I've seemed to fix it by also tracking lives.
# Now it runs for 19272 frames.)

#
# rs_tests/my_test_c2.py
# 60 seeds, small model, CPU, sequentially

#
# rs_tests/my_test_c3.py
# 6 seeds, big model, CPU, sequentially

#
# rs_tests/my_test_c4.py
# 60 seeds, big model, CPU, sequentially

#
# rs_tests/my_test_d1.py
# 6 seeds, big model, GPU, sequentially

#
# rs_tests/my_test_d2.py
# 60 seeds, big model, GPU, sequentially

# rs_tests/my_test_e1.py
# 6 seeds, big model, CPU, multiprocessing

#
# rs_tests/my_test_e2.py
# 60 seeds, big model, CPU, multiprocessing

# rs_tests/my_test_f1.py
# 6 seeds, big model, GPU, multiprocessing

#
# rs_tests/my_test_f2.py
# 60 seeds, big model, GPU, multiprocessing
