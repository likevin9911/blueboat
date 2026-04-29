/**
 * regulated_pure_pursuit_guidance_node.cpp
 *
 * Entry point. Behavior is controlled by the use_dynamic_window
 * rosparam — set it to true to enable the DWPP pass, false (default)
 * for plain RPP. Mirrors nav2's design where the same controller class
 * does both.
 */

#include <guidance/regulated_pure_pursuit_guidance.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "regulated_pure_pursuit_guidance_node");
  blueboat_coverage::RegulatedPurePursuitGuidance node;
  ros::spin();
  return 0;
}
