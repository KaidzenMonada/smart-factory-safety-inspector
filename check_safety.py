from config.dataset import COMPLIANCE_PPE_OPTIONS
from src.inference import evaluate_image
from src.model_utils import load_project_model
from src.visualization import save_prediction_image


def select_required_ppe():
    object_options = {
        str(index): class_name
        for index, class_name in enumerate(COMPLIANCE_PPE_OPTIONS, start=1)
    }

    while True:
        print("\n" + "=" * 50)
        print("Required PPE Selection")
        print("=" * 50)

        for key, value in object_options.items():
            print(f"{key}. {value.capitalize()}")

        print("\nEnter the numbers of the required PPE items.")
        print("Separate multiple selections with commas.")
        print("Example: 1,2,4")

        selection = input("\nSelection: ")

        selected_numbers = [
            item.strip()
            for item in selection.split(",")
        ]

        required_ppe = []

        for number in selected_numbers:
            if number in object_options:
                ppe_item = object_options[number]

                if ppe_item not in required_ppe:
                    required_ppe.append(ppe_item)

        if required_ppe:
            return required_ppe

        print("Please select at least one valid PPE item.")


def print_compliance_result(compliance_result):
    """
    Print a compact person-level compliance summary.
    """

    print("\nCompliance Assessment")
    print("=" * 50)
    print(f"Overall status: {compliance_result['status']}")
    print(f"Persons detected: {compliance_result['persons_detected']}")

    if compliance_result["reason"]:
        print(f"Reason: {compliance_result['reason']}")

    for person in compliance_result["persons"]:
        print(f"\nPerson {person['person_id']}: {person['status']}")
        detected_ppe = ", ".join(person["detected_ppe"]) or "none"
        print(f"Detected PPE: {detected_ppe}")

        if person["missing_ppe"]:
            print(f"PPE not detected: {', '.join(person['missing_ppe'])}")

    unassigned_ppe = compliance_result["unassigned_ppe"]

    if unassigned_ppe:
        unassigned_classes = ", ".join(
            detection["class"]
            for detection in unassigned_ppe
        )
        print(f"\nUnassigned PPE detections: {unassigned_classes}")


def main():

    # Load YOLO model
    model = load_project_model()

    # Select required PPE
    required_ppe = select_required_ppe()

    # Load image
    image_path = input("\nEnter image path: ").strip()

    # Run object detection and evaluate compliance
    evaluation = evaluate_image(
        model=model,
        image_path=image_path,
        required_ppe=required_ppe,
    )

    compliance_result = evaluation["compliance_result"]

    saved_image = save_prediction_image(
        image=evaluation["result"].orig_img,
        compliance_result=compliance_result,
        output_path="outputs/predictions",
        detections=evaluation["detections"],
    )

    print_compliance_result(compliance_result)

    print(f"\nAnnotated image saved to:")
    print(saved_image)

if __name__ == "__main__":
    main()
