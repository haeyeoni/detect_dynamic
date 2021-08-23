#define PCL_NO_PRECOMPILE
#include <ros/ros.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

struct PointXYZIA
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    int occ[50];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIA,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (int[50], occ, occ)
)

typedef PointXYZIA  PointTypeData;
typedef pcl::PointXYZI  PointType;

class DyanmicDetection
{
private:
    ros::NodeHandle nh;
    ros::Subscriber subCloud;
    ros::Publisher pubTempCloud;
    ros::Publisher pubStaticDS;
    ros::Publisher pubDynamicDS;
    ros::Publisher pubDynamic;

    pcl::PointCloud<PointTypeData>::Ptr occCloud;
    
    pcl::PointCloud<PointType>::Ptr staticCloud;
    pcl::PointCloud<PointType>::Ptr staticCloudAccum;
    pcl::PointCloud<PointType>::Ptr staticCloudDS;

    pcl::PointCloud<PointType>::Ptr dynamicCloud;
    pcl::PointCloud<PointType>::Ptr dynamicCloudAccum;
    pcl::PointCloud<PointType>::Ptr dynamicCloudDS;

    pcl::VoxelGrid<PointType> downSizeFilter;
    pcl::KdTreeFLANN<PointTypeData>::Ptr searchTreeOcc;
    pcl::KdTreeFLANN<PointType>::Ptr searchTreeOriginal;
    
    // TF 
    tf2_ros::Buffer tfbuf_;
    tf2_ros::TransformListener tfl_;
    tf2_ros::TransformBroadcaster tfb_;

    double leaf_size;
    int t_window;
    double occ_thresh;
    double x_crop, y_crop, z_crop;

    bool init = false;

public:
    DyanmicDetection():nh("~"), tfl_(tfbuf_)
    {
        nh.param<double>("leaf_size", leaf_size, 1.0); 
        nh.param<double>("occ_thresh", occ_thresh, 0.5); 
        nh.param<int>("t_window", t_window, 50); 
        nh.param<double>("x_crop", x_crop, 2.0); 
        nh.param<double>("y_crop", y_crop, 2.0); 
        nh.param<double>("z_crop", z_crop, 2.0); 
        
        occCloud.reset(new pcl::PointCloud<PointTypeData>());
        searchTreeOcc.reset(new pcl::KdTreeFLANN<PointTypeData>());
        searchTreeOriginal.reset(new pcl::KdTreeFLANN<PointType>());

        staticCloud.reset(new pcl::PointCloud<PointType>());
        dynamicCloud.reset(new pcl::PointCloud<PointType>());
        staticCloudDS.reset(new pcl::PointCloud<PointType>());
        dynamicCloudDS.reset(new pcl::PointCloud<PointType>());
        staticCloudAccum.reset(new pcl::PointCloud<PointType>());
        dynamicCloudAccum.reset(new pcl::PointCloud<PointType>());
        
        downSizeFilter.setLeafSize(leaf_size, leaf_size, leaf_size);
        
        subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &DyanmicDetection::cbCloud, this);
        pubTempCloud = nh.advertise<sensor_msgs::PointCloud2>("/transformed_cloud", 10);
        pubStaticDS = nh.advertise<sensor_msgs::PointCloud2>("/static_cloud_downsampled", 10);
        pubDynamicDS = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_cloud_downsampled", 10);
        pubDynamic = nh.advertise<sensor_msgs::PointCloud2>("/dynamic_cloud", 10);
    }


    void cbCloud(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
    {       
        pcl::PointCloud<PointType>::Ptr original_cloud(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr cropped_cloud(new pcl::PointCloud<PointType>);

        pcl::PointCloud<PointType>::Ptr downsampled_cloud(new pcl::PointCloud<PointType>);
        pcl::fromROSMsg(*cloud_msg, *original_cloud);

        // crop pointcloud

		pcl::PassThrough<PointType> pass;

		pass.setInputCloud (original_cloud);         
		pass.setFilterFieldName ("y");         
		pass.setFilterLimits (-y_crop, y_crop);    
		pass.filter (*cropped_cloud);              
																																																								
		pass.setInputCloud(cropped_cloud);
		pass.setFilterFieldName("z");           
		pass.setFilterLimits(-z_crop, z_crop);       
		pass.filter(*cropped_cloud);             

		pass.setInputCloud (cropped_cloud);      
		pass.setFilterFieldName ("x");        
		pass.setFilterLimits (-x_crop, x_crop);          
		pass.filter (*cropped_cloud);           

        // remove plane

		// Object for storing the plane model coefficients.
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
									
		pcl::SACSegmentation<PointType> seg;
		seg.setOptimizeCoefficients (true);      
		seg.setInputCloud (cropped_cloud);                 
		seg.setModelType (pcl::SACMODEL_PLANE);    
		seg.setMethodType (pcl::SAC_RANSAC);      
		seg.setMaxIterations (1000);              
		seg.setDistanceThreshold (0.1);          
		seg.segment (*inliers, *coefficients);    

        // transform point cloud to map
        Eigen::Affine3f trans_eigen;
        try 
        {
            const geometry_msgs::TransformStamped trans = tfbuf_.lookupTransform(
                "odom", "base_link", cloud_msg->header.stamp, ros::Duration(1)); 
                
            trans_eigen =  Eigen::Translation3f(trans.transform.translation.x, 
                                                                    trans.transform.translation.y, 
                                                                    trans.transform.translation.z) *
                                                Eigen::Quaternionf(trans.transform.rotation.w, 
                                                                    trans.transform.rotation.x, 
                                                                    trans.transform.rotation.y, 
                                                                    trans.transform.rotation.z);
            pcl::transformPointCloud(*cropped_cloud, *cropped_cloud, trans_eigen); //local accum: current pointcloud -> transform to the base

            sensor_msgs::PointCloud2 pointcloud_msg;    
            pcl::toROSMsg(*cropped_cloud, pointcloud_msg);
            pointcloud_msg.header = cloud_msg->header;

            pubTempCloud.publish(pointcloud_msg);
        }
        catch (tf2::TransformException& e)
        {
            ROS_INFO("Failed to transform pointcloud: %s", e.what());
            return;
        }

        downSizeFilter.setInputCloud(cropped_cloud);
        downSizeFilter.filter(*downsampled_cloud);
        std::vector<int> searchIdx;
        std::vector<float> searchDist;
        PointTypeData p_data;

        for (auto p = downsampled_cloud->begin(); p != downsampled_cloud->end(); p ++)
        {
            p_data.x = p->x;
            p_data.y = p->y;
            p_data.z = p->z;
            for (size_t i = 0; i < t_window; i ++)
                p_data.occ[i] = 0;

            if(!init) // not initiliaed 
            {
                occCloud->push_back(p_data);           
            }
            else
            {
                searchTreeOcc->setInputCloud(occCloud);
                searchTreeOcc->radiusSearch(p_data, leaf_size, searchIdx, searchDist);
                if (searchIdx.size() > 0)
                {
                    for(size_t i = 0; i < t_window -1; i ++)
                    {
                        
                        occCloud->points[searchIdx[0]].occ[i] = occCloud->points[searchIdx[0]].occ[i+1];
                    }
                    occCloud->points[searchIdx[0]].occ[t_window - 1] = 1;
                }
                else 
                {
                    occCloud->push_back(p_data);           
                }
            }
        }
        init = true;    

        // Check Dynamic

        staticCloudDS->clear();
        dynamicCloudDS->clear();           
        dynamicCloud->clear();
        
        for (auto p = occCloud->begin(); p != occCloud->end(); p ++)
        {
            int prev_sum = 0;
            double prev_avg = 0.0;
            int curr_sum = 0;
            double curr_avg = 0.0;

            for (size_t i = 0; i < t_window / 5 * 4; i ++)
                prev_sum += p->occ[i];
            prev_avg = (double) prev_sum / (double) (t_window / 5 * 4);

            for (size_t i = t_window / 5 * 4; i < t_window; i ++)
                curr_sum += p->occ[i];
            curr_avg = (double) curr_sum / (double) (t_window / 5);
            
            PointType new_p;
            new_p.x = p->x;
            new_p.y = p->y;
            new_p.z = p->z;
            // std::cout<<"prev_avg: "<<prev_avg <<" curr_avg: "<<prev_avg <<std::endl;
            if (abs(prev_avg - curr_avg) > occ_thresh)
                dynamicCloudDS->push_back(new_p);                
            else
                staticCloudDS->push_back(new_p);     
        }
        
        std::cout<<"static: "<< staticCloudDS->size()<<" dynamic: "<< dynamicCloudDS->size()
                <<" occ : "<< occCloud->size()<<" downsampled: "<<downsampled_cloud->size()<<std::endl;
        sensor_msgs::PointCloud2 dynamic_ds_msg;    
        pcl::toROSMsg(*dynamicCloudDS, dynamic_ds_msg);
        dynamic_ds_msg.header = cloud_msg->header;

        sensor_msgs::PointCloud2 static_ds_msg;    
        pcl::toROSMsg(*staticCloudDS, static_ds_msg);
        static_ds_msg.header = cloud_msg->header;

        pubDynamicDS.publish(static_ds_msg);
        pubStaticDS.publish(dynamic_ds_msg);

        // reconstruct the dynamic pointcloud

        for (auto p = dynamicCloudDS->begin(); p != dynamicCloudDS->end(); p ++)
        {
            searchTreeOriginal->setInputCloud(cropped_cloud);
            searchTreeOriginal->radiusSearch(*p, leaf_size, searchIdx, searchDist);
            for (size_t i = 0; i < searchIdx.size(); i ++)
                dynamicCloud->push_back(cropped_cloud->points[searchIdx[i]]);
        }
        std::cout<<"reconstructed: "<< dynamicCloud->size()<<std::endl;
        if(dynamicCloud->size() > 0)
        {
            // retransform the cloud            

            pcl::transformPointCloud(*dynamicCloud, *dynamicCloud, trans_eigen.inverse()); 
            sensor_msgs::PointCloud2 dynamic_msg;    
            pcl::toROSMsg(*dynamicCloud, dynamic_msg);
            dynamic_msg.header = cloud_msg->header;

            pubDynamic.publish(dynamic_msg);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detect_dynamic");
    ROS_INFO("\033[1;32m---->\033[0m Detect Dynamic Started.");
    
    DyanmicDetection DD;

    while(ros::ok()) {
         ros::spinOnce();
    } 
    return 0;
}
