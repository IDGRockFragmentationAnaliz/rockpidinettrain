def panoptic_quality(
	edges_pred: np.ndarray,
	edges_gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Выделяет замкнутые области внутри предсказанных и эталонных границ.

	В выходных масках фон и сами границы имеют значение 0, а каждая
	замкнутая область получает отдельный положительный целочисленный ID.
	"""
	if edges_pred.ndim != 2 or edges_gt.ndim != 2:
		raise ValueError("Обе карты границ должны быть двумерными")

	if edges_pred.shape != edges_gt.shape:
		raise ValueError(
			f"Размеры карт границ не совпадают: "
			f"prediction={edges_pred.shape}, GT={edges_gt.shape}"
		)

	def closed_objects(edges: np.ndarray) -> np.ndarray:
		# Ненулевые пиксели считаются непроходимыми границами. 4-связность
		# не позволяет фону просачиваться через диагональное касание линий.
		background = (edges == 0).astype(np.uint8)
		component_count, components = cv2.connectedComponents(
			background,
			connectivity=4,
		)

		border_ids = np.unique(np.concatenate((
			components[0, :],
			components[-1, :],
			components[:, 0],
			components[:, -1],
		)))

		closed_component_ids = np.setdiff1d(
			np.arange(component_count),
			np.append(border_ids, 0),
		)

		object_ids_by_component = np.zeros(component_count, dtype=np.int32)
		object_ids_by_component[closed_component_ids] = np.arange(
			1,
			len(closed_component_ids) + 1,
		)

		return object_ids_by_component[components]

	return closed_objects(edges_pred), closed_objects(edges_gt)

if __name__ == "__main__":
	main()
