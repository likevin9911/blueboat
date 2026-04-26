#include <coverage/coverage.h>

int main(int argc, char** argv){

    ros::init(argc, argv, "coverage_node");

    blueboat_navigation::Coverage coverage_node;

    ros::spin();

    return(0);
}

