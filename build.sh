cmake \
    -DUSE_PANGOLIN_VIEWER=ON \
    -DINSTALL_PANGOLIN_VIEWER=ON \
    -DUSE_SOCKET_PUBLISHER=OFF \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=ON \
    ..

# openloris mapping
./build/run_tum_rgbd_slam \
-v ./orb_vocab.fbow \
-d ../datasets/openloris/market-1-1/ \
-c example/openloris/openloris_market.yaml \
--frame-skip 1 \
--eval-log \
-p results/market1.msg

# openloris localization
./build/run_openloris_localization \
-v ./orb_vocab.fbow \
-d ../datasets/openloris/market-1-1/ \
-c example/openloris/openloris_market.yaml \
--frame-skip 1 -p results/market1.msg --eval_log


# geek+ mapping
./build/run_tum_rgbd_slam \
    -v ./orb_vocab.fbow \
    -d ../datasets/geek/map_bag/ \
    -c ./example/geek/geek.yaml -p results/geek_map.msg --eval-log --debug

# geek+ localization
./build/run_openloris_localization \
-v ./orb_vocab.fbow \
-d ../datasets/geek/relocal_bag/ \
-c ./example/geek/geek.yaml \
--frame-skip 1 -p results/geek_map.msg --debug

# change msg from openvlsam to univloc
./build/univloc_to_openvslam ~/.ros/univloc_map/geek.msg ./results/geek_map.msg

# change msg from openvlsam to univloc
./build/run_tum_rgbd_slam -v ./orb_vocab.fbow -d ../datasets/gdata0719/10-shine/ -c ./example/geek/geek_fisheye.yaml -p results/geek_fisheye_map.msg --eval-log --debug


./build/run_tum_rgbd_slam -v ./orb_vocab.fbow -d ../datasets/gdata0719/10-shine-front/ -c ./example/geek/geek.yaml -p results/geek_stereo_map.msg --eval-log --debug

./build/run_openloris_localization \
-v ./orb_vocab.fbow \
-d ../datasets/gdata0719/10-shine-front/ \
-c ./example/geek/geek.yaml \
--frame-skip 1 -p results/geek_stereo_map.msg --debug


localize_track

origin_localize

# 可以解决不同相机模型的定位