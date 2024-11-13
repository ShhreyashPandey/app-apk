from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Set up window background color and size
Window.clearcolor = (0.98, 0.98, 0.98, 1)
Window.size = (360, 640)

# Load the Flan-T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


class ChatbotApp(App):
    def build(self):
        # Main layout as vertical
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Header section
        header_label = Label(
            text="FOOD AI",
            font_size='24sp',
            color=(0, 0, 0, 1),
            bold=True,
            size_hint=(1, 0.1)
        )
        main_layout.add_widget(header_label)

        # Image upload section with camera icon
        upload_box = BoxLayout(orientation='vertical', size_hint=(1, 0.4), padding=20, spacing=10)
        self.camera_icon = Image(source="camera_icon.png",
                                 allow_stretch=True)  # Replace with the path to your camera icon image
        upload_button = Button(
            text="Upload Image",
            font_size='18sp',
            size_hint=(1, 0.5),
            background_normal='',  # Remove background to use color
            background_color=(0.9, 0.9, 0.9, 1),
            color=(0, 0, 0, 1)
        )
        upload_button.bind(on_release=self.open_file_chooser)
        upload_box.add_widget(self.camera_icon)
        upload_box.add_widget(upload_button)
        main_layout.add_widget(upload_box)

        # Button to analyze the image
        analyze_button = Button(
            text="Analyze the Image",
            font_size='18sp',
            background_color=(0.3, 0.6, 0.9, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, 0.1)
        )
        analyze_button.bind(on_press=self.analyze_image)
        main_layout.add_widget(analyze_button)

        # Prediction label with a background container
        prediction_box = BoxLayout(size_hint=(1, 0.15), padding=10)
        self.prediction_label = Label(
            text="The image is of...",
            font_size='16sp',
            color=(0, 0, 0, 1),
            halign="center",
            valign="middle"
        )
        self.prediction_label.bind(size=self.prediction_label.setter('text_size'))
        prediction_box.add_widget(self.prediction_label)
        main_layout.add_widget(prediction_box)

        # Chatbot question buttons
        chatbot_layout = BoxLayout(orientation='vertical', padding=(0, 10), spacing=10)

        # Button to ask nutritional analysis
        nutrition_button = Button(
            text="Nutritional Analysis of pancake?",
            font_size='16sp',
            background_color=(0.3, 0.6, 0.9, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, 0.1)
        )
        nutrition_button.bind(on_press=self.ask_nutritional_info)
        chatbot_layout.add_widget(nutrition_button)

        # Button to ask how to make it
        how_to_make_button = Button(
            text="How to make them?",
            font_size='16sp',
            background_color=(0.3, 0.6, 0.9, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, 0.1)
        )
        how_to_make_button.bind(on_press=self.ask_how_to_make)
        chatbot_layout.add_widget(how_to_make_button)

        # Chatbot response label with background container
        chatbot_response_box = BoxLayout(size_hint=(1, 0.15), padding=10)
        self.chatbot_label = Label(
            text="Chatbot Response: ",
            font_size='16sp',
            color=(0, 0, 0, 1),
            halign="center",
            valign="middle"
        )
        self.chatbot_label.bind(size=self.chatbot_label.setter('text_size'))
        chatbot_response_box.add_widget(self.chatbot_label)
        chatbot_layout.add_widget(chatbot_response_box)

        main_layout.add_widget(chatbot_layout)

        # Bottom question input for custom questions
        question_input_box = BoxLayout(size_hint=(1, 0.1), padding=(0, 5), spacing=5)
        self.question_input = TextInput(
            hint_text="Type...",
            font_size='16sp',
            background_color=(0.9, 0.9, 0.9, 1),
            foreground_color=(0, 0, 0, 1),
            size_hint=(0.85, 1),
            multiline=False
        )
        send_button = Button(
            text="📷",
            font_size='20sp',
            background_color=(0.8, 0.8, 0.8, 1),
            color=(0, 0, 0, 1),
            size_hint=(0.15, 1)
        )
        send_button.bind(on_press=self.send_question)
        question_input_box.add_widget(self.question_input)
        question_input_box.add_widget(send_button)

        main_layout.add_widget(question_input_box)

        return main_layout

    def open_file_chooser(self, instance):
        # Open file chooser to upload an image
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserIconView()
        content.add_widget(file_chooser)

        select_button = Button(text="Select Image", size_hint=(1, 0.2))
        content.add_widget(select_button)

        popup = Popup(title="Select an Image", content=content, size_hint=(0.9, 0.9))
        select_button.bind(on_press=lambda x: self.load_image(file_chooser.selection, popup))
        popup.open()

    def load_image(self, selection, popup):
        # Load selected image and close popup
        if selection:
            self.camera_icon.source = selection[0]
        popup.dismiss()

    def analyze_image(self, instance):
        # Placeholder function to analyze the uploaded image
        self.prediction_label.text = "The image is of pancakes."

    def ask_nutritional_info(self, instance):
        # Placeholder function for nutritional info
        self.chatbot_label.text = "Calories: 312 Kcal\nTotal Fat: 4g\nCarbohydrates: 41g"

    def ask_how_to_make(self, instance):
        # Placeholder function for how to make it
        self.chatbot_label.text = "Here is how you make them: [Steps...]"

    def send_question(self, instance):
        # Function to send a custom question to the Flan-T5 model
        question = self.question_input.text
        if question:
            # Prepare the input for the Flan-T5 model
            input_text = f"Answer the question: {question}"
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate response using the Flan-T5 model
            outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Display the response
            self.chatbot_label.text = f"Chatbot Response: {response}"
            self.question_input.text = ""  # Clear input after sending


if __name__ == "__main__":
    ChatbotApp().run()
