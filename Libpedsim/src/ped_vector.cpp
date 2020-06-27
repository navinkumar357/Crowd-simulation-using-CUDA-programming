//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// 
// Adapted for Low Level Parallel Programming 2017

#define _USE_MATH_DEFINES
#include "ped_vector.h"

#include <cmath>
#include <string>
#include <sstream>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

/// Default constructor, which makes sure that all the values are set to 0.
/// \date    2012-01-16
Ped::Tvector::Tvector() : x(0), y(0), z(0) {};


std::string Ped::Tvector::to_string() const {
	double x_ = x;
	double y_ = y;
	double z_ = z;
	std::ostringstream strs;
	strs << x << "/" << y << "/" << z;
	return strs.str();
	//return std::to_string((const long double) x) + "/" + std::to_string((const long double) y_) + "/" + std::to_string((const long double) z_);
}


/// Returns the length of the vector.
/// \return the length
double Ped::Tvector::length() const {
	if ((x == 0) && (y == 0) && (z == 0)) return 0;
	return sqrt(lengthSquared());
}


/// Returns the length of the vector squared. This is faster than the real length.
/// \return the length squared
double Ped::Tvector::lengthSquared() const {
	return x*x + y*y + z*z;
}


/// Normalizes the vector to a length of 1.
/// \date    2010-02-12
void Ped::Tvector::normalize() {
	double len = length();

	// null vectors cannot be normalized
	if (len == 0)
		return;

	x /= len;
	y /= len;
	z /= len;
}


/// Normalizes the vector to a length of 1.
/// \date    2013-08-02
Ped::Tvector Ped::Tvector::normalized() const {
	double len = length();

	// null vectors cannot be normalized
	if (len == 0)
		return Ped::Tvector();;

	return Ped::Tvector(x / len, y / len, z / len);
}


/// Vector scalar product helper: calculates the scalar product of two vectors.
/// \date    2012-01-14
/// \return  The scalar product.
/// \param   &a The first vector
/// \param   &b The second vector
double Ped::Tvector::scalar(const Ped::Tvector &a, const Ped::Tvector &b) {
	return acos(Ped::Tvector::dotProduct(a, b) / (a.length() * b.length()));
}


/// Vector scalar product helper: calculates the scalar product of two vectors.
/// \date    2012-01-14
/// \return  The scalar product.
/// \param   &a The first vector
/// \param   &b The second vector
double Ped::Tvector::dotProduct(const Ped::Tvector &a, const Ped::Tvector &b) {
	return (a.x*b.x + a.y*b.y + a.z*b.z);
}


/// Calculates the cross product of two vectors.
/// \date    2010-02-12
/// \param   &a The first vector
/// \param   &b The second vector
Ped::Tvector Ped::Tvector::crossProduct(const Ped::Tvector &a, const Ped::Tvector &b) {
	return Ped::Tvector(
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x);
}


/// Scales this vector by a given factor in each dimension.
/// \date    2013-08-02
/// \param   factor The scalar value to multiply with.
void Ped::Tvector::scale(double factor) {
	x *= factor;
	y *= factor;
	z *= factor;
}


/// Returns a copy of this vector which is multiplied in each dimension by a given factor.
/// \date    2013-07-16
/// \return  The scaled vector.
/// \param   factor The scalar value to multiply with.
Ped::Tvector Ped::Tvector::scaled(double factor) const {
	return Ped::Tvector(factor*x, factor*y, factor*z);
}

Ped::Tvector Ped::Tvector::leftNormalVector() const {
	return Ped::Tvector(-y, x);
}

Ped::Tvector Ped::Tvector::rightNormalVector() const {
	return Ped::Tvector(y, -x);
}

double Ped::Tvector::polarRadius() const {
	return length();
}

double Ped::Tvector::polarAngle() const {
	return atan2(y, x);
}

double Ped::Tvector::angleTo(const Tvector &other) const {
	double angleThis = polarAngle();
	double angleOther = other.polarAngle();

	// compute angle
	double diffAngle = angleOther - angleThis;
	// → normalize angle
	if (diffAngle > M_PI)
		diffAngle -= 2 * M_PI;
	else if (diffAngle <= -M_PI)
		diffAngle += 2 * M_PI;

	return diffAngle;
}

Ped::Tvector Ped::Tvector::operator+(const Tvector& other) const {
	return Ped::Tvector(
		x + other.x,
		y + other.y,
		z + other.z);
}

Ped::Tvector Ped::Tvector::operator-(const Tvector& other) const {
	return Ped::Tvector(
		x - other.x,
		y - other.y,
		z - other.z);
}

Ped::Tvector Ped::Tvector::operator*(double factor) const {
	return scaled(factor);
}

Ped::Tvector Ped::Tvector::operator/(double divisor) const {
	return scaled(1 / divisor);
}

Ped::Tvector& Ped::Tvector::operator+=(const Tvector& vectorIn) {
	x += vectorIn.x;
	y += vectorIn.y;
	z += vectorIn.z;
	return *this;
}

Ped::Tvector& Ped::Tvector::operator-=(const Tvector& vectorIn) {
	x -= vectorIn.x;
	y -= vectorIn.y;
	z -= vectorIn.z;
	return *this;
}

Ped::Tvector& Ped::Tvector::operator*=(double factor) {
	scale(factor);
	return *this;
}

Ped::Tvector& Ped::Tvector::operator*=(const Tvector& vectorIn) {
	x *= vectorIn.x;
	y *= vectorIn.y;
	z *= vectorIn.z;
	return *this;
}

Ped::Tvector& Ped::Tvector::operator/=(double divisor) {
	scale(1 / divisor);
	return *this;
}

bool operator==(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In) {
	return (vector1In.x == vector2In.x)
		&& (vector1In.y == vector2In.y)
		&& (vector1In.z == vector2In.z);
}

bool operator!=(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In) {
	return (vector1In.x != vector2In.x)
		|| (vector1In.y != vector2In.y)
		|| (vector1In.z != vector2In.z);
}

Ped::Tvector operator+(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In) {
	return Ped::Tvector(
		vector1In.x + vector2In.x,
		vector1In.y + vector2In.y,
		vector1In.z + vector2In.z);
}

Ped::Tvector operator-(const Ped::Tvector& vector1In, const Ped::Tvector& vector2In) {
	return Ped::Tvector(
		vector1In.x - vector2In.x,
		vector1In.y - vector2In.y,
		vector1In.z - vector2In.z);
}

Ped::Tvector operator-(const Ped::Tvector& vectorIn) {
	return Ped::Tvector(
		-vectorIn.x,
		-vectorIn.y,
		-vectorIn.z);
}

Ped::Tvector operator*(double factor, const Ped::Tvector& vector) {
	return vector.scaled(factor);
}
