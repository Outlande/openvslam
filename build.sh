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
-d ../datasets/openloris/market-1-1/ \
-c example/openloris/openloris_market.yaml \
--frame-skip 1 \
--eval-log \
-p results/market1.msg


./build/run_openloris_localization \
-v ./orb_vocab.fbow \
-d ../datasets/openloris/market-1-1/ \
-c example/openloris/openloris_market.yaml \
--frame-skip 1 -p results/market1.msg


./build/run_tum_rgbd_slam \
    -v ./orb_vocab.fbow \
    -d ../datasets/geek/map_bag/ \
    -c ./example/geek/geek.yaml -p results/geek_map.msg --eval-log --debug


./build/run_openloris_localization \
-v ./orb_vocab.fbow \
-d ../datasets/geek/relocal_bag/ \
-c ./example/geek/geek.yaml \
--frame-skip 1 -p results/geek_map.msg --debug

./build/univloc_to_openvslam ./orb_vocab.fbow example/geek/geek.yaml ./results/geek_map.msg


# 是否开mapping模块
projection match 中搜索的金字塔level会影响定位成功率，原来只search下一层，现在每一层都search，search的level多，效果就会更好
(可能不同level的匹配会有问题)

判断是否lost

不更新描述子

不优化inliner

rotaion 和 translation