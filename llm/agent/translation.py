from util.mylog import logger
from typing import Optional, Tuple
from langchain.base_language import BaseLanguageModel

# 开源项目：https://github.com/andrewyng/translation-agent

class TranslationAgent:
    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm
    
    def one_chunk_initial_translation(
        self, source_lang, target_lang, source_text
    ):
        """
        Translate the entire text as one chunk using an LLM.

        Args:
            source_lang (str): The source language of the text.
            target_lang (str): The target language for translation.
            source_text (str): The text to be translated.

        Returns:
            str: The translated text.
        """

        system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
        translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

        prompt = translation_prompt.format(source_text=source_text)
        return (prompt, system_message)

    def one_chunk_reflect_on_translation(
        self,
        source_lang,
        target_lang,
        source_text,
        translation_1,
        country = "",
    ):
        """
        Use an LLM to reflect on the translation, treating the entire text as one chunk.

        Args:
            source_lang (str): The source language of the text.
            target_lang (str): The target language of the translation.
            source_text (str): The original text in the source language.
            translation_1 (str): The initial translation of the source text.
            country (str): Country specified for target language.

        Returns:
            str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
        """

        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

        if country != "":
            reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

        else:
            reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

        prompt = reflection_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            translation_1=translation_1,
        )
        return (prompt, system_message)


    def one_chunk_improve_translation(
        self,
        source_lang,
        target_lang,
        source_text,
        translation_1,
        reflection,
    ):
        """
        Use the reflection to improve the translation, treating the entire text as one chunk.

        Args:
            source_lang (str): The source language of the text.
            target_lang (str): The target language for the translation.
            source_text (str): The original text in the source language.
            translation_1 (str): The initial translation of the source text.
            reflection (str): Expert suggestions and constructive criticism for improving the translation.

        Returns:
            str: The improved translation based on the expert suggestions.
        """

        system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

        prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""
        
        return (prompt, system_message)

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str = "",
        llm_chat = None
    ) -> str:
        """
        Translate a single chunk of text from the source language to the target language.

        This function performs a two-step translation process:
        1. Get an initial translation of the source text.
        2. Reflect on the initial translation and generate an improved translation.

        Args:
            source_lang (str): The source language of the text.
            target_lang (str): The target language for the translation.
            source_text (str): The text to be translated.
            country (str): Country specified for target language.
            llm_chat: The LLM chat interface to use for translation.
        Returns:
            str: The improved translation of the source text.
        """
        if not llm_chat:
            raise Exception("llm_chat init invalid")
        
        translation_prompt_1 = self.one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )
        llm_chat.appendSystemInfo(translation_prompt_1[1])
        logger.info("====== one_chunk_initial_translation") 
        isok, translation_1 = llm_chat.chat(translation_prompt_1[0])
        if not isok:
            raise Exception("one_chunk_initial_translation invalid")

        reflection_prompt = self.one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, translation_1, country
        )
        llm_chat.appendSystemInfo(reflection_prompt[1])
        logger.info("====== one_chunk_reflect_on_translation") 
        isok, reflection = llm_chat.chat(reflection_prompt[0])  # 修正这里的错误，使用正确的提示
        if not isok:
            raise Exception("one_chunk_reflect_on_translation invalid")

        translation_prompt_2 = self.one_chunk_improve_translation(
            source_lang, target_lang, source_text, translation_1, reflection
        )
        llm_chat.appendSystemInfo(translation_prompt_2[1])
        logger.info("====== one_chunk_improve_translation") 
        isok, translation_2 = llm_chat.chat(translation_prompt_2[0])
        if not isok:
            raise Exception("one_chunk_improve_translation invalid")

        return translation_2


if __name__ == "__main__":
    from llmapi.llm_factory import LLMFactory, LLMChatAdapter
    # 初始化翻译代理
    llm = LLMFactory.create("siliconflow", model_name="deepseek-ai/DeepSeek-V3")
    translator = TranslationAgent(llm)
    
    # 创建LLM适配器
    llm_chat_adapter = LLMChatAdapter(llm)
    
    # 测试翻译功能
    test_text = "Hello world! This is a test of the translation agent."
    
    try:
        result = translator.translate(
            source_lang="English",
            target_lang="Chinese",
            source_text=test_text,
            llm_chat=llm_chat_adapter
        )
        logger.info(f"原文: {test_text}")
        logger.info(f"翻译结果: {result}")
    except Exception as e:
        logger.error(f"翻译过程中出错: {str(e)}")