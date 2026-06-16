#include "CGLTFMeshFileLoader.h"
#include "CMeshBuffer.h"
#include "coreutil.h"
#include "IAnimatedMesh.h"
#include "ILogger.h"
#include "IReadFile.h"
#include "irrTypes.h"
#include "os.h"
#include "path.h"
#include "S3DVertex.h"
#include "SAnimatedMesh.h"
#include "SColor.h"
#include "SMesh.h"
#include "vector3d.h"
#include "matrix4.h"
#include "quaternion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

#include <cstddef>
#include <cstring>
#include <cmath>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/* Notes on the coordinate system.
 *
 * glTF uses a right-handed coordinate system where +Z is the
 * front-facing axis, and Irrlicht uses a left-handed coordinate
 * system where -Z is the front-facing axis.
 * We convert between them by reflecting the mesh across the X axis.
 * Doing this correctly requires negating the Z coordinate on
 * vertex positions and normals, and reversing the winding order
 * of the vertex indices.
 */

static const float SKELETAL_FPS = 24.0f;

static bool FakeImageLoader(
	tinygltf::Image* img,
	const int image_idx,
	std::string* err,
	std::string* warn,
	int req_width,
	int req_height,
	const unsigned char* bytes,
	int size,
	void* user_data)
{
	return true;
}

namespace irr {

	namespace scene {

		CGLTFMeshFileLoader::BufferOffset::BufferOffset(
			const std::vector<unsigned char>& buf,
			const std::size_t offset)
			: m_buf(buf)
			, m_offset(offset)
		{
		}

		CGLTFMeshFileLoader::BufferOffset::BufferOffset(
			const CGLTFMeshFileLoader::BufferOffset& other,
			const std::size_t fromOffset)
			: m_buf(other.m_buf)
			, m_offset(other.m_offset + fromOffset)
		{
		}

		/**
		 * Get a raw unsigned char (ubyte) from a buffer offset.
		*/
		unsigned char CGLTFMeshFileLoader::BufferOffset::at(
			const std::size_t fromOffset) const
		{
			return m_buf.at(m_offset + fromOffset);
		}

		CGLTFMeshFileLoader::CGLTFMeshFileLoader() noexcept
		{
		}

		/**
		 * The most basic portion of the code base. This tells irllicht if this file has a .gltf extension.
		*/
		bool CGLTFMeshFileLoader::isALoadableFileExtension(
			const io::path& filename) const
		{
			return core::hasFileExtension(filename, "gltf");
		}

		/**
		 * Entry point into loading a GLTF model.
		*/
		IAnimatedMesh* CGLTFMeshFileLoader::createMesh(io::IReadFile* file)
		{
			tinygltf::Model model{};

			if (file->getSize() <= 0 || !tryParseGLTF(file, model)) {
				return nullptr;
			}

			MeshExtractor parser(std::move(model));

			SAnimatedMesh* animatedMesh(new SAnimatedMesh{});

			if (parser.hasSkin()) {
				// Skeletal animation: bake each frame into a static SMesh.
				// Frame 0 is the base pose (time=0), subsequent frames are
				// baked at 1/24s intervals across all concatenated animations.
				loadSkeletalFrames(parser, animatedMesh);
			}
			else {
				// No skeleton: load base mesh + morph targets as before.
				SMesh* baseMesh(new SMesh{});
				loadPrimitives(parser, baseMesh);
				baseMesh->recalculateBoundingBox();
				animatedMesh->addMesh(baseMesh); // frame 0 = base pose
				baseMesh->drop();
				loadMorphTargets(parser, animatedMesh); // frames 1..N
			}

			animatedMesh->setAnimationSpeed(0);
			animatedMesh->recalculateBoundingBox();
			return animatedMesh;
		}


		/**
		 * Load up the rawest form of the model. The vertex positions and indices.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes
		 * If material is undefined, then a default material MUST be used.
		*/
		void CGLTFMeshFileLoader::loadPrimitives(
			const MeshExtractor& parser,
			SMesh* mesh)
		{
			for (std::size_t i = 0; i < parser.getMeshCount(); ++i) {
				for (std::size_t j = 0; j < parser.getPrimitiveCount(i); ++j) {
					auto indices = parser.getIndices(i, j);
					auto vertices = parser.getVertices(i, j);

					SMeshBufferLightMap* meshbuf(new SMeshBufferLightMap{});
					meshbuf->append(vertices.data(), vertices.size(),
						indices.data(), indices.size());
					mesh->addMeshBuffer(meshbuf);
					meshbuf->drop();
				}
			}
		}

		void CGLTFMeshFileLoader::loadMorphTargets(
			const MeshExtractor& parser,
			SAnimatedMesh* animatedMesh)
		{
			for (std::size_t i = 0; i < parser.getMeshCount(); ++i) {
				for (std::size_t j = 0; j < parser.getPrimitiveCount(i); ++j) {
					const std::size_t targetCount = parser.getMorphTargetCount(i, j);
					if (targetCount == 0)
						continue;

					auto indices = parser.getIndices(i, j);

					for (std::size_t t = 0; t < targetCount; ++t) {
						auto vertices = parser.getMorphTargetVertices(i, j, t);

						SMesh* frameMesh(new SMesh{});
						SMeshBufferLightMap* meshbuf(new SMeshBufferLightMap{});
						meshbuf->append(vertices.data(), vertices.size(),
							indices.data(), indices.size());
						frameMesh->addMeshBuffer(meshbuf);
						meshbuf->drop();
						frameMesh->recalculateBoundingBox();
						animatedMesh->addMesh(frameMesh);
						frameMesh->drop();
					}
				}
			}
		}

		CGLTFMeshFileLoader::MeshExtractor::MeshExtractor(
			const tinygltf::Model& model) noexcept
			: m_model(model)
		{
		}

		CGLTFMeshFileLoader::MeshExtractor::MeshExtractor(
			const tinygltf::Model&& model) noexcept
			: m_model(model)
		{
		}

		/**
		 * Extracts GLTF mesh indices into the irrlicht model.
		*/
		std::vector<u16> CGLTFMeshFileLoader::MeshExtractor::getIndices(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto accessorIdx = getIndicesAccessorIdx(meshIdx, primitiveIdx);
			const auto& buf = getBuffer(accessorIdx);

			std::vector<u16> indices{};
			const auto count = getElemCount(accessorIdx);
			for (std::size_t i = 0; i < count; ++i) {
				std::size_t elemIdx = count - i - 1;
				indices.push_back(readPrimitive<u16>(
					BufferOffset(buf, elemIdx * sizeof(u16))));
			}

			return indices;
		}

		/**
		 * Create a vector of video::S3DVertex (model data) from a mesh & primitive index.
		*/
		std::vector<CGLTFMeshFileLoader::MeshExtractor::vertex_t> CGLTFMeshFileLoader::MeshExtractor::getVertices(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto positionAccessorIdx = getPositionAccessorIdx(
				meshIdx, primitiveIdx);
			std::vector<vertex_t> vertices{};
			vertices.resize(getElemCount(positionAccessorIdx));
			for (auto& v : vertices) {
				v.Color = video::SColor(255, 255, 255, 255);
			}
			copyPositions(positionAccessorIdx, vertices);

			const auto normalAccessorIdx = getNormalAccessorIdx(
				meshIdx, primitiveIdx);
			if (normalAccessorIdx != static_cast<std::size_t>(-1)) {
				copyNormals(normalAccessorIdx, vertices);
			}

			const auto tCoordAccessorIdx = getTCoordAccessorIdx(
				meshIdx, primitiveIdx);
			if (tCoordAccessorIdx != static_cast<std::size_t>(-1)) {
				copyTCoords(tCoordAccessorIdx, vertices);
			}

			const auto tCoord1AccessorIdx = getTCoord1AccessorIdx(
				meshIdx, primitiveIdx);
			if (tCoord1AccessorIdx != static_cast<std::size_t>(-1)) {
				copyTCoords2(tCoord1AccessorIdx, vertices);
			}

			const auto colorAccessorIdx = getColorAccessorIdx(
				meshIdx, primitiveIdx);
			if (colorAccessorIdx != static_cast<std::size_t>(-1)) {
				copyColors(colorAccessorIdx, vertices);
			}

			return vertices;
		}

		/**
		 * Get the amount of meshes that a model contains.
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getMeshCount() const
		{
			return m_model.meshes.size();
		}

		/**
		 * Get the amount of primitives that a mesh in a model contains.
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getPrimitiveCount(
			const std::size_t meshIdx) const
		{
			return m_model.meshes[meshIdx].primitives.size();
		}

		/**
		 * Templated buffer reader. Based on type width.
		 * This is specifically used to build upon to read more complex data types.
		 * It is also used raw to read arrays directly.
		 * Basically we're using the width of the type to infer
		 * how big of a gap we have from the beginning of the buffer.
		*/
		template <typename T>
		T CGLTFMeshFileLoader::MeshExtractor::readPrimitive(
			const BufferOffset& readFrom)
		{
			unsigned char d[sizeof(T)]{};
			for (std::size_t i = 0; i < sizeof(T); ++i) {
				d[i] = readFrom.at(i);
			}
			T dest;
			std::memcpy(&dest, d, sizeof(dest));
			return dest;
		}

		/**
		 * Read a vector2df from a buffer at an offset.
		 * @return vec2 core::Vector2df
		*/
		core::vector2df CGLTFMeshFileLoader::MeshExtractor::readVec2DF(
			const CGLTFMeshFileLoader::BufferOffset& readFrom)
		{
			return core::vector2df(readPrimitive<float>(readFrom),
				readPrimitive<float>(BufferOffset(readFrom, sizeof(float))));

		}

		/**
		 * Read a vector3df from a buffer at an offset.
		 * @return vec3 core::Vector3df
		*/
		core::vector3df CGLTFMeshFileLoader::MeshExtractor::readVec3DF(
			const BufferOffset& readFrom,
			const core::vector3df scale = { 1.0f,1.0f,1.0f })
		{
			return core::vector3df(
				scale.X * readPrimitive<float>(readFrom),
				scale.Y * readPrimitive<float>(BufferOffset(readFrom, sizeof(float))),
				-scale.Z * readPrimitive<float>(BufferOffset(readFrom, 2 *
					sizeof(float))));
		}

		/**
		 * Streams vertex positions raw data into usable buffer via reference.
		 * Buffer: ref Vector<video::S3DVertex>
		*/
		void CGLTFMeshFileLoader::MeshExtractor::copyPositions(
			const std::size_t accessorIdx,
			std::vector<vertex_t>& vertices) const
		{

			const auto& buffer = getBuffer(accessorIdx);
			const auto count = getElemCount(accessorIdx);
			const auto byteStride = getByteStride(accessorIdx);

			for (std::size_t i = 0; i < count; i++) {
				const auto v = readVec3DF(BufferOffset(buffer,
					(byteStride * i)), getScale());
				vertices[i].Pos = v;
			}
		}

		/**
		 * Streams normals raw data into usable buffer via reference.
		 * Buffer: ref Vector<video::S3DVertex>
		*/
		void CGLTFMeshFileLoader::MeshExtractor::copyNormals(
			const std::size_t accessorIdx,
			std::vector<vertex_t>& vertices) const
		{
			const auto& buffer = getBuffer(accessorIdx);
			const auto count = getElemCount(accessorIdx);
			const auto byteStride = getByteStride(accessorIdx);

			for (std::size_t i = 0; i < count; i++) {
				const auto n = readVec3DF(BufferOffset(buffer,
					(byteStride * i)));
				vertices[i].Normal = n;
			}
		}

		/**
		 * Streams texture coordinate raw data into usable buffer via reference.
		 * Buffer: ref Vector<video::S3DVertex>
		*/
		void CGLTFMeshFileLoader::MeshExtractor::copyTCoords(
			const std::size_t accessorIdx,
			std::vector<vertex_t>& vertices) const
		{

			const auto& buffer = getBuffer(accessorIdx);
			const auto count = getElemCount(accessorIdx);
			const auto byteStride = getByteStride(accessorIdx);

			for (std::size_t i = 0; i < count; ++i) {
				const auto t = readVec2DF(BufferOffset(buffer,
					(byteStride * i)));
				vertices[i].TCoords = t;
			}
		}

		/**
		 * Streams texture coordinate (set 1 / lightmap) raw data into usable buffer via reference.
		 * Buffer: ref Vector<video::S3DVertex2TCoords>
		*/
		void CGLTFMeshFileLoader::MeshExtractor::copyTCoords2(
			const std::size_t accessorIdx,
			std::vector<vertex_t>& vertices) const
		{
			const auto& buffer = getBuffer(accessorIdx);
			const auto count = getElemCount(accessorIdx);
			const auto byteStride = getByteStride(accessorIdx);

			for (std::size_t i = 0; i < count; ++i) {
				const auto t = readVec2DF(BufferOffset(buffer,
					(byteStride * i)));
				vertices[i].TCoords2 = t;
			}
		}

		/**
		 * Streams vertex color raw data into usable buffer via reference.
		 * COLOR_0 is VEC4 FLOAT with components in [0, 1].
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		*/
		void CGLTFMeshFileLoader::MeshExtractor::copyColors(
			const std::size_t accessorIdx,
			std::vector<vertex_t>& vertices) const
		{
			const auto& buffer = getBuffer(accessorIdx);
			const auto count = getElemCount(accessorIdx);
			const auto byteStride = getByteStride(accessorIdx);
			const auto& accessor = m_model.accessors[accessorIdx];

			for (std::size_t i = 0; i < count; ++i) {
				const BufferOffset base(buffer, byteStride * i);
				u8 r, g, b, a;

				if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
					r = static_cast<u8>(readPrimitive<u16>(BufferOffset(base, 0 * sizeof(u16))) / 257);
					g = static_cast<u8>(readPrimitive<u16>(BufferOffset(base, 1 * sizeof(u16))) / 257);
					b = static_cast<u8>(readPrimitive<u16>(BufferOffset(base, 2 * sizeof(u16))) / 257);
					a = static_cast<u8>(readPrimitive<u16>(BufferOffset(base, 3 * sizeof(u16))) / 257);
				}
				else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
					r = readPrimitive<u8>(BufferOffset(base, 0));
					g = readPrimitive<u8>(BufferOffset(base, 1));
					b = readPrimitive<u8>(BufferOffset(base, 2));
					a = readPrimitive<u8>(BufferOffset(base, 3));
				}
				else {
					// FLOAT
					r = static_cast<u8>(readPrimitive<float>(BufferOffset(base, 0 * sizeof(float))) * 255.0f);
					g = static_cast<u8>(readPrimitive<float>(BufferOffset(base, 1 * sizeof(float))) * 255.0f);
					b = static_cast<u8>(readPrimitive<float>(BufferOffset(base, 2 * sizeof(float))) * 255.0f);
					a = static_cast<u8>(readPrimitive<float>(BufferOffset(base, 3 * sizeof(float))) * 255.0f);
				}

				vertices[i].Color = video::SColor(a, r, g, b);
			}
		}

		/**
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-node
		 * Type: number[3] (tinygltf: vector<double>)
		 * Required: NO
		 * @returns: core::vector2df
		*/
		core::vector3df CGLTFMeshFileLoader::MeshExtractor::getScale() const
		{
			core::vector3df buffer{ 1.0f,1.0f,1.0f };
			if (!m_model.nodes.empty() && m_model.nodes[0].scale.size() == 3) {
				buffer.X = static_cast<float>(m_model.nodes[0].scale[0]);
				buffer.Y = static_cast<float>(m_model.nodes[0].scale[1]);
				buffer.Z = static_cast<float>(m_model.nodes[0].scale[2]);
			}
			return buffer;
		}

		/**
		 * The number of elements referenced by this accessor, not to be confused with the number of bytes or number of components.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_count
		 * Type: Integer
		 * Required: YES
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getElemCount(
			const std::size_t accessorIdx) const
		{
			return m_model.accessors[accessorIdx].count;
		}

		/**
		 * The stride, in bytes, between vertex attributes.
		 * When this is not defined, data is tightly packed.
		 * When two or more accessors use the same buffer view, this field MUST be defined.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_bufferview_bytestride
		 * Required: NO
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getByteStride(
			const std::size_t accessorIdx) const
		{
			const auto& accessor = m_model.accessors[accessorIdx];
			const auto& view = m_model.bufferViews[accessor.bufferView];
			return accessor.ByteStride(view);
		}

		/**
		 * Specifies whether integer data values are normalized (true) to [0, 1] (for unsigned types)
		 * or to [-1, 1] (for signed types) when they are accessed. This property MUST NOT be set to
		 * true for accessors with FLOAT or UNSIGNED_INT component type.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_accessor_normalized
		 * Required: NO
		*/
		bool CGLTFMeshFileLoader::MeshExtractor::isAccessorNormalized(
			const std::size_t accessorIdx) const
		{
			const auto& accessor = m_model.accessors[accessorIdx];
			return accessor.normalized;
		}

		/**
		 * Walk through the complex chain of the model to extract the required buffer.
		 * Accessor -> BufferView -> Buffer
		*/
		CGLTFMeshFileLoader::BufferOffset CGLTFMeshFileLoader::MeshExtractor::getBuffer(
			const std::size_t accessorIdx) const
		{
			const auto& accessor = m_model.accessors[accessorIdx];
			const auto& view = m_model.bufferViews[accessor.bufferView];
			const auto& buffer = m_model.buffers[view.buffer];

			return BufferOffset(buffer.data, view.byteOffset);
		}

		/**
		 * The index of the accessor that contains the vertex indices.
		 * When this is undefined, the primitive defines non-indexed geometry.
		 * When defined, the accessor MUST have SCALAR type and an unsigned integer component type.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_mesh_primitive_indices
		 * Type: Integer
		 * Required: NO
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getIndicesAccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			return m_model.meshes[meshIdx].primitives[primitiveIdx].indices;
		}

		/**
		 * The index of the accessor that contains the POSITIONs.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		 * Type: VEC3 (Float)
		 * ! Required: YES (Appears so, needs another pair of eyes to research.)
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getPositionAccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			return m_model.meshes[meshIdx].primitives[primitiveIdx]
				.attributes.find("POSITION")->second;
		}

		/**
		 * Get if a model contains animation.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation
		 * Type: vector<Animation>
		 * Required: NO
		*/
		bool CGLTFMeshFileLoader::MeshExtractor::isAnimated() const {
			return m_model.animations.size() > 0;
		}

		/**
		 * The index of the accessor that contains the NORMALs.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		 * Type: VEC3 (Float)
		 * ! Required: NO (Appears to not be, needs another pair of eyes to research.)
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getNormalAccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto& attributes = m_model.meshes[meshIdx]
				.primitives[primitiveIdx].attributes;
			const auto result = attributes.find("NORMAL");

			if (result == attributes.end()) {
				return -1;
			}
			else {
				return result->second;
			}
		}

		/**
		 * The index of the accessor that contains the NORMALs.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		 * Type: VEC3 (Float)
		 * ! Required: YES (Appears so, needs another pair of eyes to research.)
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getTCoordAccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto& attributes = m_model.meshes[meshIdx]
				.primitives[primitiveIdx].attributes;
			const auto result = attributes.find("TEXCOORD_0");

			if (result == attributes.end()) {
				return -1;
			}
			else {
				return result->second;
			}
		}

		/**
		 * The index of the accessor that contains TEXCOORD_1 (lightmap UVs).
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		 * Type: VEC2 (Float)
		 * Required: NO
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getTCoord1AccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto& attributes = m_model.meshes[meshIdx]
				.primitives[primitiveIdx].attributes;
			const auto result = attributes.find("TEXCOORD_1");

			if (result == attributes.end()) {
				return -1;
			}
			else {
				return result->second;
			}
		}

		/**
		 * The index of the accessor that contains vertex colors (COLOR_0).
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		 * Type: VEC4 (Float)
		 * Required: NO
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getColorAccessorIdx(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			const auto& attributes = m_model.meshes[meshIdx]
				.primitives[primitiveIdx].attributes;
			const auto result = attributes.find("COLOR_0");

			if (result == attributes.end()) {
				return -1;
			}
			else {
				return result->second;
			}
		}

		/**
		 * Get the number of morph targets for a given mesh primitive.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#morph-targets
		*/
		std::size_t CGLTFMeshFileLoader::MeshExtractor::getMorphTargetCount(
			const std::size_t meshIdx,
			const std::size_t primitiveIdx) const
		{
			return m_model.meshes[meshIdx].primitives[primitiveIdx].targets.size();
		}

		/**
		 * Build a vertex array for a morph target frame by applying POSITION
		 * and NORMAL deltas (if present) onto the base mesh vertices.
		 *
		 * glTF morph targets store deltas, not absolute values.
		 * The same Z-negation coordinate flip used in readVec3DF is applied.
		 * Documentation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#morph-targets
		*/
		std::vector<CGLTFMeshFileLoader::MeshExtractor::vertex_t>
			CGLTFMeshFileLoader::MeshExtractor::getMorphTargetVertices(
				const std::size_t meshIdx,
				const std::size_t primitiveIdx,
				const std::size_t targetIdx) const
		{
			// Start from the fully-populated base vertices (positions, normals,
			// UVs, colors) so non-positional attributes are preserved.
			auto vertices = getVertices(meshIdx, primitiveIdx);

			const auto& target = m_model.meshes[meshIdx]
				.primitives[primitiveIdx].targets[targetIdx];

			// Apply POSITION deltas
			const auto posIt = target.find("POSITION");
			if (posIt != target.end()) {
				const std::size_t accessorIdx = posIt->second;
				const auto& buffer = getBuffer(accessorIdx);
				const auto count = getElemCount(accessorIdx);
				const auto byteStride = getByteStride(accessorIdx);
				const auto scale = getScale();

				for (std::size_t i = 0; i < count && i < vertices.size(); ++i) {
					const auto delta = readVec3DF(
						BufferOffset(buffer, byteStride * i), scale);
					vertices[i].Pos += delta;
				}
			}

			// Apply NORMAL deltas
			const auto normIt = target.find("NORMAL");
			if (normIt != target.end()) {
				const std::size_t accessorIdx = normIt->second;
				const auto& buffer = getBuffer(accessorIdx);
				const auto count = getElemCount(accessorIdx);
				const auto byteStride = getByteStride(accessorIdx);

				for (std::size_t i = 0; i < count && i < vertices.size(); ++i) {
					const auto delta = readVec3DF(
						BufferOffset(buffer, byteStride * i));
					vertices[i].Normal += delta;
					vertices[i].Normal.normalize();
				}
			}

			return vertices;
		}


		bool CGLTFMeshFileLoader::MeshExtractor::hasSkin() const
		{
			return !m_model.skins.empty();
		}

		int CGLTFMeshFileLoader::MeshExtractor::getTotalSkeletalFrameCount() const
		{
			float totalSeconds = 0.0f;
			for (const auto& anim : m_model.animations) {
				float animMax = 0.0f;
				for (const auto& channel : anim.channels) {
					const auto& sampler = anim.samplers[channel.sampler];
					const auto& accessor = m_model.accessors[sampler.input];
					const auto& view = m_model.bufferViews[accessor.bufferView];
					const float* times = reinterpret_cast<const float*>(
						m_model.buffers[view.buffer].data.data()
						+ view.byteOffset + accessor.byteOffset);
					if (accessor.count > 0) {
						float last = times[accessor.count - 1];
						if (last > animMax) animMax = last;
					}
				}
				totalSeconds += animMax;
			}
			return static_cast<int>(std::ceil(totalSeconds * SKELETAL_FPS));
		}

		static core::vector3df sampleVec3(
			const tinygltf::Model& model,
			const int accessorIdx,
			const std::size_t idx)
		{
			const auto& accessor = model.accessors[accessorIdx];
			const auto& view = model.bufferViews[accessor.bufferView];
			const float* data = reinterpret_cast<const float*>(
				model.buffers[view.buffer].data.data()
				+ view.byteOffset + accessor.byteOffset);
			return core::vector3df(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]);
		}

		static core::quaternion sampleQuat(
			const tinygltf::Model& model,
			const int accessorIdx,
			const std::size_t idx)
		{
			const auto& accessor = model.accessors[accessorIdx];
			const auto& view = model.bufferViews[accessor.bufferView];
			const float* data = reinterpret_cast<const float*>(
				model.buffers[view.buffer].data.data()
				+ view.byteOffset + accessor.byteOffset);
			return core::quaternion(
				-data[idx * 4 + 3],
				-data[idx * 4],
				-data[idx * 4 + 1],
				-data[idx * 4 + 2]);
		}

		static core::vector3df lerpVec3(
			const core::vector3df& a, const core::vector3df& b, float t)
		{
			return a + (b - a) * t;
		}

		static void findKeyframeBlend(
			const tinygltf::Model& model,
			const int inputAccessorIdx,
			const float time,
			std::size_t& outIdx,
			float& outT)
		{
			const auto& accessor = model.accessors[inputAccessorIdx];
			const auto& view = model.bufferViews[accessor.bufferView];
			const float* times = reinterpret_cast<const float*>(
				model.buffers[view.buffer].data.data()
				+ view.byteOffset + accessor.byteOffset);
			const std::size_t count = accessor.count;

			if (count == 0) { outIdx = 0; outT = 0.0f; return; }
			if (time <= times[0]) { outIdx = 0; outT = 0.0f; return; }
			if (time >= times[count - 1]) { outIdx = count - 1; outT = 0.0f; return; }

			for (std::size_t i = 0; i < count - 1; ++i) {
				if (time >= times[i] && time < times[i + 1]) {
					outIdx = i;
					outT = (time - times[i]) / (times[i + 1] - times[i]);
					return;
				}
			}
			outIdx = count - 1;
			outT = 0.0f;
		}

		std::vector<core::matrix4> CGLTFMeshFileLoader::MeshExtractor::buildJointMatrices(
			float timeSeconds) const
		{
			const auto& skin = m_model.skins[0];
			const std::size_t jointCount = skin.joints.size();
			const std::size_t nodeCount = m_model.nodes.size();

			std::vector<std::array<float, 3>> T(nodeCount, { 0,0,0 });
			std::vector<std::array<float, 4>> R(nodeCount, { 0,0,0,1 });
			std::vector<std::array<float, 3>> S(nodeCount, { 1,1,1 });

			for (std::size_t i = 0; i < nodeCount; ++i) {
				const auto& node = m_model.nodes[i];
				if (node.translation.size() == 3)
					T[i] = { (float)node.translation[0], (float)node.translation[1], (float)node.translation[2] };
				if (node.rotation.size() == 4)
					R[i] = { (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2], (float)node.rotation[3] };
				if (node.scale.size() == 3)
					S[i] = { (float)node.scale[0], (float)node.scale[1], (float)node.scale[2] };
			}

			float localTime = timeSeconds;
			for (const auto& anim : m_model.animations) {
				float animDuration = 0.0f;
				for (const auto& channel : anim.channels) {
					const auto& sampler = anim.samplers[channel.sampler];
					const auto& acc = m_model.accessors[sampler.input];
					const auto& view = m_model.bufferViews[acc.bufferView];
					const float* times = reinterpret_cast<const float*>(
						m_model.buffers[view.buffer].data.data() + view.byteOffset + acc.byteOffset);
					if (acc.count > 0 && times[acc.count - 1] > animDuration)
						animDuration = times[acc.count - 1];
				}

				if (localTime <= animDuration || &anim == &m_model.animations.back()) {
					for (const auto& channel : anim.channels) {
						if (channel.target_node < 0) continue;
						const std::size_t ni = (std::size_t)channel.target_node;
						const auto& sampler = anim.samplers[channel.sampler];
						const bool isStep = sampler.interpolation == "STEP";

						std::size_t kIdx = 0; float blend = 0.0f;
						findKeyframeBlend(m_model, sampler.input, localTime, kIdx, blend);
						if (isStep) blend = 0.0f;

						const auto& outAcc = m_model.accessors[sampler.output];
						const auto& outView = m_model.bufferViews[outAcc.bufferView];
						const float* outData = reinterpret_cast<const float*>(
							m_model.buffers[outView.buffer].data.data() + outView.byteOffset + outAcc.byteOffset);

						if (channel.target_path == "translation") {
							const float* a = outData + kIdx * 3;
							if (blend > 0.0f) {
								const float* b = outData + (kIdx + 1) * 3;
								T[ni] = { a[0] + (b[0] - a[0]) * blend, a[1] + (b[1] - a[1]) * blend, a[2] + (b[2] - a[2]) * blend };
							}
							else {
								T[ni] = { a[0], a[1], a[2] };
							}
						}
						else if (channel.target_path == "rotation") {
							const float* a = outData + kIdx * 4;
							if (blend > 0.0f) {
								const float* b = outData + (kIdx + 1) * 4;
								float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
								float s = dot < 0.0f ? -1.0f : 1.0f;
								float qa[4] = { a[0], a[1], a[2], a[3] };
								float qb[4] = { s * b[0], s * b[1], s * b[2], s * b[3] };
								float r[4];
								for (int k = 0; k < 4; ++k) r[k] = qa[k] + (qb[k] - qa[k]) * blend;
								float len = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
								if (len > 0.0f) for (int k = 0; k < 4; ++k) r[k] /= len;
								R[ni] = { r[0], r[1], r[2], r[3] };
							}
							else {
								R[ni] = { a[0], a[1], a[2], a[3] };
							}
						}
						else if (channel.target_path == "scale") {
							const float* a = outData + kIdx * 3;
							if (blend > 0.0f) {
								const float* b = outData + (kIdx + 1) * 3;
								S[ni] = { a[0] + (b[0] - a[0]) * blend, a[1] + (b[1] - a[1]) * blend, a[2] + (b[2] - a[2]) * blend };
							}
							else {
								S[ni] = { a[0], a[1], a[2] };
							}
						}
					}
					break;
				}
				localTime -= animDuration;
			}

			auto trsToMatrix = [](const std::array<float, 3>& t,
				const std::array<float, 4>& r,
				const std::array<float, 3>& s) -> core::matrix4 {
					float x = r[0], y = r[1], z = r[2], w = r[3];
					core::matrix4 m;
					m[0] = (1 - 2 * (y * y + z * z)) * s[0]; m[4] = 2 * (x * y - z * w) * s[1]; m[8] = 2 * (x * z + y * w) * s[2]; m[12] = t[0];
					m[1] = 2 * (x * y + z * w) * s[0]; m[5] = (1 - 2 * (x * x + z * z)) * s[1]; m[9] = 2 * (y * z - x * w) * s[2]; m[13] = t[1];
					m[2] = 2 * (x * z - y * w) * s[0]; m[6] = 2 * (y * z + x * w) * s[1]; m[10] = (1 - 2 * (x * x + y * y)) * s[2]; m[14] = t[2];
					m[3] = 0;                     m[7] = 0;                     m[11] = 0;                     m[15] = 1;
					return m;
				};

			std::vector<int> parent(nodeCount, -1);
			for (std::size_t i = 0; i < nodeCount; ++i)
				for (int child : m_model.nodes[i].children)
					parent[child] = (int)i;

			std::vector<core::matrix4> globalTransforms(nodeCount);
			std::vector<bool> resolved(nodeCount, false);
			bool anyResolved = true;
			while (anyResolved) {
				anyResolved = false;
				for (std::size_t i = 0; i < nodeCount; ++i) {
					if (resolved[i]) continue;
					if (parent[i] != -1 && !resolved[parent[i]]) continue;
					core::matrix4 local = trsToMatrix(T[i], R[i], S[i]);
					globalTransforms[i] = (parent[i] == -1)
						? local
						: globalTransforms[parent[i]] * local;
					resolved[i] = true;
					anyResolved = true;
				}
			}

			std::vector<core::matrix4> inverseBindMatrices(jointCount);
			if (skin.inverseBindMatrices >= 0) {
				const auto& acc = m_model.accessors[skin.inverseBindMatrices];
				const auto& view = m_model.bufferViews[acc.bufferView];
				const float* data = reinterpret_cast<const float*>(
					m_model.buffers[view.buffer].data.data() + view.byteOffset + acc.byteOffset);
				for (std::size_t j = 0; j < jointCount; ++j) {
					core::matrix4& m = inverseBindMatrices[j];
					for (int i = 0; i < 16; ++i)
						m[i] = data[j * 16 + i];
				}
			}

			std::vector<core::matrix4> jointMatrices(jointCount);
			for (std::size_t j = 0; j < jointCount; ++j)
				jointMatrices[j] = globalTransforms[skin.joints[j]] * inverseBindMatrices[j];

			return jointMatrices;
		}

		std::vector<CGLTFMeshFileLoader::MeshExtractor::vertex_t>
			CGLTFMeshFileLoader::MeshExtractor::getSkinnedVertices(
				const std::size_t meshIdx,
				const std::size_t primitiveIdx,
				const std::vector<core::matrix4>& jointMatrices) const
		{
			auto vertices = getVertices(meshIdx, primitiveIdx);
			const auto& primitive = m_model.meshes[meshIdx].primitives[primitiveIdx];

			const auto jointsIt = primitive.attributes.find("JOINTS_0");
			const auto weightsIt = primitive.attributes.find("WEIGHTS_0");
			if (jointsIt == primitive.attributes.end()
				|| weightsIt == primitive.attributes.end())
				return vertices;

			const auto& jointsAcc = m_model.accessors[jointsIt->second];
			const auto& weightsAcc = m_model.accessors[weightsIt->second];
			const auto& jointsView = m_model.bufferViews[jointsAcc.bufferView];
			const auto& weightsView = m_model.bufferViews[weightsAcc.bufferView];

			const std::size_t count = jointsAcc.count;
			const std::size_t jointsStride = jointsAcc.ByteStride(jointsView);
			const std::size_t weightsStride = weightsAcc.ByteStride(weightsView);

			const unsigned char* jointsData =
				m_model.buffers[jointsView.buffer].data.data()
				+ jointsView.byteOffset + jointsAcc.byteOffset;
			const unsigned char* weightsData =
				m_model.buffers[weightsView.buffer].data.data()
				+ weightsView.byteOffset + weightsAcc.byteOffset;

			const auto posAccessorIdx = getPositionAccessorIdx(meshIdx, primitiveIdx);
			const auto& posBuf = getBuffer(posAccessorIdx);
			const auto posStride = getByteStride(posAccessorIdx);
			const auto scale = getScale();
			const auto normalAccessorIdx = getNormalAccessorIdx(meshIdx, primitiveIdx);
			const bool hasNormals = normalAccessorIdx != static_cast<std::size_t>(-1);

			const unsigned char* normalData = nullptr;
			std::size_t normalStride = 0;
			if (hasNormals) {
				const auto& nAcc = m_model.accessors[normalAccessorIdx];
				const auto& nView = m_model.bufferViews[nAcc.bufferView];
				normalData = m_model.buffers[nView.buffer].data.data()
					+ nView.byteOffset + nAcc.byteOffset;
				normalStride = nAcc.ByteStride(nView);
			}

			for (std::size_t i = 0; i < count && i < vertices.size(); ++i) {
				u16 joints[4] = {};
				if (jointsAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
					const unsigned char* j = jointsData + jointsStride * i;
					joints[0] = j[0]; joints[1] = j[1];
					joints[2] = j[2]; joints[3] = j[3];
				}
				else {
					const u16* j = reinterpret_cast<const u16*>(
						jointsData + jointsStride * i);
					joints[0] = j[0]; joints[1] = j[1];
					joints[2] = j[2]; joints[3] = j[3];
				}

				const float* w = reinterpret_cast<const float*>(
					weightsData + weightsStride * i);

				const BufferOffset posOff(posBuf, posStride * i);
				core::vector3df rawPos(
					scale.X * readPrimitive<float>(posOff),
					scale.Y * readPrimitive<float>(BufferOffset(posOff, sizeof(float))),
					scale.Z * readPrimitive<float>(BufferOffset(posOff, 2 * sizeof(float))));

				core::vector3df skinnedPos(0, 0, 0);
				core::vector3df skinnedNormal(0, 0, 0);

				core::vector3df rawNormal(0, 0, 0);
				if (hasNormals && normalData) {
					const float* nf = reinterpret_cast<const float*>(
						normalData + normalStride * i);
					rawNormal = core::vector3df(nf[0], nf[1], nf[2]);
				}

				for (int k = 0; k < 4; ++k) {
					if (w[k] == 0.0f) continue;
					if (joints[k] >= jointMatrices.size()) continue;
					const auto& jm = jointMatrices[joints[k]];
					core::vector3df p = rawPos;
					jm.transformVect(p);
					skinnedPos += p * w[k];
					if (hasNormals) {
						core::vector3df n = rawNormal;
						jm.rotateVect(n);
						skinnedNormal += n * w[k];
					}
				}

				vertices[i].Pos = core::vector3df(skinnedPos.X, skinnedPos.Y, -skinnedPos.Z);
				if (hasNormals) {
					skinnedNormal.normalize();
					vertices[i].Normal = core::vector3df(skinnedNormal.X, skinnedNormal.Y, -skinnedNormal.Z);
				}
			}

			return vertices;
		}

		void CGLTFMeshFileLoader::loadSkeletalFrames(
			const MeshExtractor& parser,
			SAnimatedMesh* animatedMesh)
		{
			const int totalFrames = parser.getTotalSkeletalFrameCount();

			for (int frame = 0; frame <= totalFrames; ++frame) {
				const float time = frame / SKELETAL_FPS;
				const auto jointMatrices = parser.buildJointMatrices(time);

				SMesh* frameMesh(new SMesh{});
				for (std::size_t i = 0; i < parser.getMeshCount(); ++i) {
					for (std::size_t j = 0; j < parser.getPrimitiveCount(i); ++j) {
						auto indices = parser.getIndices(i, j);
						auto vertices = parser.getSkinnedVertices(i, j, jointMatrices);

						SMeshBufferLightMap* meshbuf(new SMeshBufferLightMap{});
						meshbuf->append(vertices.data(), vertices.size(),
							indices.data(), indices.size());
						frameMesh->addMeshBuffer(meshbuf);
						meshbuf->drop();
					}
				}
				frameMesh->recalculateBoundingBox();
				animatedMesh->addMesh(frameMesh);
				frameMesh->drop();
			}
		}


		/**
		 * This is where the actual model's GLTF file is loaded and parsed by tinygltf.
		*/
		bool CGLTFMeshFileLoader::tryParseGLTF(io::IReadFile* file,
			tinygltf::Model& model)
		{
			tinygltf::TinyGLTF loader{};

			loader.SetImageLoader(FakeImageLoader, nullptr);

			std::string warn;
			std::string err;
			const long size = file->getSize();

			auto buf = std::make_unique<char[]>(size);
			long readBytes = file->read(buf.get(), size);

			io::path filename = file->getFileName();
			s32 slash1 = filename.findLast('/');
			s32 slash2 = filename.findLast('\\');
			s32 slash = slash1 > slash2 ? slash1 : slash2;
			io::path baseDir = (slash >= 0) ? filename.subString(0, slash + 1) : "";

			bool ok = loader.LoadASCIIFromString(
				&model,
				&err,
				&warn,
				buf.get(),
				size,
				baseDir.c_str(),
				1
			);

			if (!warn.empty()) os::Printer::log(warn.c_str(), ELL_WARNING);
			if (!err.empty()) os::Printer::log(err.c_str(), ELL_ERROR);

			return ok;
		}

	} // namespace scene

} // namespace irr