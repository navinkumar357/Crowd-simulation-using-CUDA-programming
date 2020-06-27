//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//

#ifndef _ped_vector_h_
#define _ped_vector_h_ 1

#define DllExport __declspec(dllexport)

#include <string>

namespace Ped {
	/// Vector helper class. This is basically a struct with some related functions attached.
	/// x, y, and z are public, so that they can be accessed easily.
	/// \author  chgloor
	/// \date    2010-02-12
	class DllExport Tvector {
	public:
		// Default constructor
		Tvector();

		// Initializing constructor
		Tvector(double px, double py, double pz = 0) : x(px), y(py), z(pz) {};


		// Methods
		double length() const;
		double lengthSquared() const;
		void normalize();
		Tvector normalized() const;
		void scale(double factor);
		Tvector scaled(double factor) const;

		Tvector leftNormalVector() const;
		Tvector rightNormalVector() const;

		double polarRadius() const;
		double polarAngle() const;

		double angleTo(const Tvector &other) const;

		static double scalar(const Tvector &a, const Tvector &b);
		static double dotProduct(const Tvector &a, const Tvector &b);
		static Tvector crossProduct(const Tvector &a, const Tvector &b);

		std::string to_string() const;


		// Operators
		Tvector operator+(const Tvector& other) const;
		Tvector operator-(const Tvector& other) const;
		Tvector operator*(double factor) const;
		Tvector operator/(double divisor) const;
		Tvector& operator+=(const Tvector& vectorIn);
		Tvector& operator-=(const Tvector& vectorIn);
		Tvector& operator*=(double factor);
		Tvector& operator*=(const Tvector& vectorIn);
		Tvector& operator/=(double divisor);


		// Attributes
		double x;
		double y;
		double z;
	};
}

bool operator==(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In);
bool operator!=(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In);
Ped::Tvector operator-(const Ped::Tvector& vectorIn);
Ped::Tvector operator*(double factor, const Ped::Tvector& vector);

#endif
