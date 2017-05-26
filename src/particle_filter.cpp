/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	is_initialized = true;
	num_particles = 50; // TODO: Hyper Parameter

	weights.clear();
	weights.reserve(num_particles);
	particles.clear();
	particles.reserve(num_particles);

	std::default_random_engine gen;
	std::normal_distribution<double> N_x(0, std[0]);
	std::normal_distribution<double> N_y(0, std[1]);
	std::normal_distribution<double> N_theta(0, std[2]);

	for (int i=0; i<num_particles; ++i){
		Particle p;
		p.id = i;
		p.x = x + N_x(gen);
		p.y = y + N_y(gen);
		p.theta = fmod(theta + N_theta(gen), 2.0 * M_PI);
		p.weight = 1;
		particles.push_back(p);
		weights.push_back(1);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	std::default_random_engine gen;
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);

	if (fabs(yaw_rate) < 1.0e-06){
		for (int i=0; i<num_particles; ++i){
			double x_diff = velocity * delta_t * cos(particles[i].theta);
			double y_diff = velocity * delta_t * sin(particles[i].theta);
			particles[i].x +=  x_diff;
			particles[i].y +=  y_diff;

			// add noise
			particles[i].x += N_x(gen);
			particles[i].y += N_y(gen);
			particles[i].theta = fmod(particles[i].theta + N_theta(gen), 2.0 * M_PI);

			particles[i].id = i;
		}
	} else {
		for (int i=0; i<num_particles; ++i){
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;

      //add noise
      particles[i].x += N_x(gen);
      particles[i].y += N_y(gen);
      particles[i].theta = fmod(particles[i].theta + N_theta(gen), 2.0 * M_PI);

      particles[i].id = i;
		}
	}
}

void transformCarToMap(Particle p, std::vector<LandmarkObs>& observations){
	for (int i=0; i<observations.size(); ++i){
		double map_x = cos(p.theta) * observations[i].x - sin(p.theta) * observations[i].y + p.x;
		double map_y = sin(p.theta) * observations[i].x + cos(p.theta) * observations[i].y + p.y;
		observations[i].x = map_x;
		observations[i].y = map_y;
	}
}

double calDistance(double p_x, double p_y, double g_x, double g_y){
	return sqrt((p_x - g_x) * (p_x - g_x) + (p_y - g_y) * (p_y - g_y));
}

void filterCandicate(Particle p,  double sensor_range, Map map_landmarks, std::vector<LandmarkObs>& predicted){
	for (int i=0; i<map_landmarks.landmark_list.size(); ++i){
			double dist = calDistance(p.x, p.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);
			if (dist < sensor_range){
				LandmarkObs obs;
				obs.x = map_landmarks.landmark_list[i].x_f;
				obs.y = map_landmarks.landmark_list[i].y_f;
				obs.id = map_landmarks.landmark_list[i].id_i;
				predicted.push_back(obs);
			}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution.
	double weights_sum = 0.0;

	for (int i=0; i<num_particles; ++i){
		Particle p = particles[i];

	  std::vector<LandmarkObs> transObservations(observations);
		transformCarToMap(p, transObservations);

		std::vector<LandmarkObs> predicted;
		filterCandicate(p, sensor_range, map_landmarks, predicted);

		// Select Nearest Neighbors point
		double weight = 1.0;
		for (int j=0; j<transObservations.size(); ++j){
			double min_dist = sensor_range;
			LandmarkObs nearest_landmark;
			bool found = false;
			int pred_size = predicted.size();
			for (int k=0; k<pred_size; ++k){
				double dist = calDistance(transObservations[j].x, transObservations[j].y,
					predicted[k].x, predicted[k].y);
				if (dist < min_dist){
					nearest_landmark = predicted[k];
					min_dist = dist;
					found = true;
				}
			}

			// Update particle weight
      double update_x, update_y;

      if (found) {
        update_x = (transObservations[j].x - nearest_landmark.x);
        update_y = (transObservations[j].y - nearest_landmark.y);
      } else {
        update_x = sensor_range;
        update_y = sensor_range;
      }

      update_x = update_x * update_x;
      update_x /= 2 * std_landmark[0] * std_landmark[0];
      update_y = update_y * update_y;
      update_y /= 2 * std_landmark[1] * std_landmark[1];
      double ex = -1.0 * (update_x + update_y);
      double prob = (1.0 / sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]) ) * exp(ex);
      weight *= prob;
		}

		weights_sum += weight;

		p.weight = weight;
		weights[i] = weight;

		particles[i] = p;

	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	std::vector<Particle> new_particles;
	new_particles.reserve(num_particles);

	std::default_random_engine gen;
	std::discrete_distribution<> d(weights.begin(), weights.end());

	for (int i=0; i<num_particles; ++i){
		new_particles[i] = particles[d(gen)];
	}

	for (int i=0; i<num_particles; ++i){
		particles[i] = new_particles[i];
	}
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
