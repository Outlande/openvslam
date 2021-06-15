cmake \
    -DUSE_PANGOLIN_VIEWER=ON \
    -DINSTALL_PANGOLIN_VIEWER=ON \
    -DUSE_SOCKET_PUBLISHER=OFF \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=ON \
    ..

./build/run_tum_rgbd_slam \
    -v ./orb_vocab.fbow \
    -d ../datasets/geek/map_bag/ \
    -c ./example/geek/geek.yaml -p results/geek_map.msg --eval-log


./build/run_geek_localization \
-v ./orb_vocab.fbow \
-d ../datasets/geek/relocal_bag/ \
-c ./example/geek/geek.yaml \
--frame-skip 1 -p results/geek_map.msg --debug


# 是否开mapping模块