# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# BENCHMARKS
# python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m3
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m4
# python ../yaml_to_sh.py sweeps/ENGD.yaml sweeps/ENGD.sh --qos=m4

# WOODBURY ENGD
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_exact.yaml sweeps/ENGD_woodbury_exact.sh --qos=m3 --array=24
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_naive.yaml sweeps/ENGD_woodbury_naive.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_nystrom.yaml sweeps/ENGD_woodbury_nystrom.sh --qos=m3 --array=50


# SPRING
python ../yaml_to_sh.py sweeps/SPRING_exact.yaml sweeps/SPRING_exact.sh --qos=m4 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_naive.yaml sweeps/SPRING_naive.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom.yaml sweeps/SPRING_nystrom.sh --qos=m3 --array=50