from __future__ import annotations

import os
import ctypes
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_int32,
    c_uint8,
    c_uint32,
    c_float,
    c_void_p,
    c_size_t,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    byref,
)
import pathlib
from typing import (
    List,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

import llama_cpp.llama_cpp as llama_cpp

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp.llama_types import (
        llama_token,
        llama_pos,
    )
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
        CtypesPointer,
    )

# Define input text structure
class mtmd_input_text(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]

# Define context parameters structure
class mtmd_context_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("verbosity", c_int),
        ("image_marker", c_char_p),  # const char*
        ("media_marker", c_char_p),  # const char*
    ]

# Define input chunk type enum
mtmd_input_chunk_type = c_int
(
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
) = (0, 1, 2)

# Define slice template enum
mtmd_slice_tmpl = c_int
(
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
) = (0, 1, 2, 3)

# Define whisper filters structure
class whisper_filters(Structure):
    _fields_ = [
        ("n_mel", c_int),
    ]

# Define mtmd_context structure
class mtmd_context(Structure):
    _fields_ = [
        ("ctx_v", c_void_p),  # clip_ctx*
        ("ctx_a", c_void_p),  # clip_ctx*
        ("text_model", c_void_p),  # const llama_model*
        ("image_embd_v", POINTER(c_float)),  # std::vector<float>
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("media_marker", c_char_p),  # std::string
        ("n_embd_text", c_int),
        ("img_beg", c_char_p),  # std::string
        ("img_end", c_char_p),  # std::string
        ("aud_beg", c_char_p),  # std::string
        ("aud_end", c_char_p),  # std::string
        ("slice_tmpl", c_int),  # mtmd_slice_tmpl
        ("tok_ov_img_start", llama_cpp.llama_token),
        ("tok_ov_img_end", llama_cpp.llama_token),
        ("tok_slices_start", llama_cpp.llama_token),
        ("tok_slices_end", llama_cpp.llama_token),
        ("tok_sli_img_start", llama_cpp.llama_token),
        ("tok_sli_img_end", llama_cpp.llama_token),
        ("tok_sli_img_mid", llama_cpp.llama_token),
        ("tok_row_end", llama_cpp.llama_token),
        ("tok_row_end_trail", c_bool),
        ("ov_img_first", c_bool),
        ("use_mrope", c_bool),
        ("w_filters", whisper_filters),
    ]

# Define bitmap structure
class mtmd_bitmap(Structure):
    _fields_ = [
        ("nx", c_uint32),
        ("ny", c_uint32),
        ("data", POINTER(c_uint8)),  # Vector represented as pointer
        ("id", c_char_p),
        ("is_audio", c_bool),
    ]

# Define image tokens structure
class mtmd_image_tokens(Structure):
    _fields_ = [
        ("nx", c_uint32),
        ("ny", c_uint32),
        ("use_mrope_pos", c_bool),
        ("batch_f32", c_void_p),  # clip_image_f32_batch
        ("id", c_char_p),
    ]

# Define audio tokens structure
class mtmd_audio_tokens(Structure):
    _fields_ = [
        ("n_tokens", c_uint32),
        ("batch_f32", c_void_p),  # clip_image_f32_batch
        ("id", c_char_p),
    ]

# Define input chunk structure
class mtmd_input_chunk(Structure):
    _fields_ = [
        ("type", mtmd_input_chunk_type),
        ("tokens_text", POINTER(llama_cpp.llama_token)),  # Vector represented as pointer
        ("tokens_image", c_void_p),  # mtmd_image_tokens_ptr
        ("tokens_audio", c_void_p),  # mtmd_audio_tokens_ptr
    ]

# Define input chunks structure
class mtmd_input_chunks(Structure):
    _fields_ = [
        ("entries", POINTER(mtmd_input_chunk)),  # Vector represented as pointer
    ]

# Define context pointer type
mtmd_context_p = NewType("mtmd_context_p", int)
mtmd_context_p_ctypes = c_void_p

# Define bitmap pointer type
mtmd_bitmap_p = NewType("mtmd_bitmap_p", int)
mtmd_bitmap_p_ctypes = c_void_p

# Define input chunks pointer type
mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p

# Define input chunk pointer type
mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

# Define image tokens pointer type
mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

# Define audio tokens pointer type
mtmd_audio_tokens_p = NewType("mtmd_audio_tokens_p", int)
mtmd_audio_tokens_p_ctypes = c_void_p

# Load the library
_libmtmd_base_name = "mtmd"
_libmtmd_override_path = os.environ.get("MTMD_CPP_LIB")
_libmtmd_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libmtmd_override_path is None else pathlib.Path()

# Load the library
_libmtmd = load_shared_library(_libmtmd_base_name, _libmtmd_base_path)
ctypes_function = ctypes_function_for_shared_library(_libmtmd)

################################################
# mtmd.h types
################################################

# Opaque types
mtmd_context_p = NewType("mtmd_context_p", int)
mtmd_context_p_ctypes = c_void_p

mtmd_bitmap_p = NewType("mtmd_bitmap_p", int)
mtmd_bitmap_p_ctypes = c_void_p

mtmd_image_tokens_p = NewType("mtmd_image_tokens_p", int)
mtmd_image_tokens_p_ctypes = c_void_p

mtmd_input_chunk_p = NewType("mtmd_input_chunk_p", int)
mtmd_input_chunk_p_ctypes = c_void_p

mtmd_input_chunks_p = NewType("mtmd_input_chunks_p", int)
mtmd_input_chunks_p_ctypes = c_void_p

# Enums
MTMD_INPUT_CHUNK_TYPE_TEXT = 0
MTMD_INPUT_CHUNK_TYPE_IMAGE = 1
MTMD_INPUT_CHUNK_TYPE_AUDIO = 2

# Structures
class mtmd_context_params(Structure):
    _fields_ = [
        ("use_gpu", c_bool),
        ("print_timings", c_bool),
        ("n_threads", c_int),
        ("verbosity", c_int),  # ggml_log_level
        ("image_marker", c_char_p),
        ("media_marker", c_char_p),
    ]

class mtmd_input_text(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("add_special", c_bool),
        ("parse_special", c_bool),
    ]

################################################
# mtmd.h functions
################################################

# MTMD_API const char * mtmd_default_marker(void);
@ctypes_function("mtmd_default_marker", [], c_char_p)
def mtmd_default_marker() -> bytes:
    ...

# MTMD_API struct mtmd_context_params mtmd_context_params_default(void);
@ctypes_function("mtmd_context_params_default", [], mtmd_context_params)
def mtmd_context_params_default() -> mtmd_context_params:
    ...

# MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
#                                             const struct llama_model * text_model,
#                                             const struct mtmd_context_params ctx_params);
@ctypes_function(
    "mtmd_init_from_file",
    [c_char_p, llama_cpp.llama_model_p_ctypes, mtmd_context_params],
    mtmd_context_p_ctypes
)
def mtmd_init_from_file(
    mmproj_fname: bytes,
    text_model: llama_cpp.llama_model_p,
    ctx_params: mtmd_context_params,
    /,
) -> Optional[mtmd_context_p]:
    ...

# MTMD_API void mtmd_free(mtmd_context * ctx);
@ctypes_function("mtmd_free", [mtmd_context_p_ctypes], None)
def mtmd_free(ctx: mtmd_context_p, /):
    ...

# MTMD_API bool mtmd_support_vision(mtmd_context * ctx);
@ctypes_function("mtmd_support_vision", [mtmd_context_p_ctypes], c_bool)
def mtmd_support_vision(ctx: mtmd_context_p, /) -> bool:
    ...

# MTMD_API mtmd_bitmap * mtmd_bitmap_init(uint32_t nx, uint32_t ny, const unsigned char * data);
@ctypes_function(
    "mtmd_bitmap_init",
    [c_uint32, c_uint32, POINTER(c_uint8)],
    mtmd_bitmap_p_ctypes
)
def mtmd_bitmap_init(
    nx: Union[c_uint32, int],
    ny: Union[c_uint32, int],
    data: CtypesArray[c_uint8],
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

# MTMD_API void mtmd_bitmap_free(mtmd_bitmap * bitmap);
@ctypes_function("mtmd_bitmap_free", [mtmd_bitmap_p_ctypes], None)
def mtmd_bitmap_free(bitmap: mtmd_bitmap_p, /):
    ...

# MTMD_API mtmd_input_chunks * mtmd_input_chunks_init(void);
@ctypes_function("mtmd_input_chunks_init", [], mtmd_input_chunks_p_ctypes)
def mtmd_input_chunks_init() -> Optional[mtmd_input_chunks_p]:
    ...

# MTMD_API void mtmd_input_chunks_free(mtmd_input_chunks * chunks);
@ctypes_function("mtmd_input_chunks_free", [mtmd_input_chunks_p_ctypes], None)
def mtmd_input_chunks_free(chunks: mtmd_input_chunks_p, /):
    ...

# MTMD_API size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
@ctypes_function("mtmd_input_chunks_size", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_input_chunks_size(chunks: mtmd_input_chunks_p, /) -> int:
    ...

# MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx);
@ctypes_function(
    "mtmd_input_chunks_get",
    [mtmd_input_chunks_p_ctypes, c_size_t],
    mtmd_input_chunk_p_ctypes
)
def mtmd_input_chunks_get(
    chunks: mtmd_input_chunks_p, idx: Union[c_size_t, int], /
) -> Optional[mtmd_input_chunk_p]:
    ...

# MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
#                                mtmd_input_chunks * output,
#                                const mtmd_input_text * text,
#                                const mtmd_bitmap ** bitmaps,
#                                size_t n_bitmaps);
@ctypes_function(
    "mtmd_tokenize",
    [
        mtmd_context_p_ctypes,
        mtmd_input_chunks_p_ctypes,
        POINTER(mtmd_input_text),
        POINTER(mtmd_bitmap_p_ctypes),
        c_size_t,
    ],
    c_int,
)
def mtmd_tokenize(
    ctx: mtmd_context_p,
    output: mtmd_input_chunks_p,
    text: "_Pointer[mtmd_input_text]",
    bitmaps: CtypesArray[mtmd_bitmap_p_ctypes],
    n_bitmaps: Union[c_size_t, int],
    /,
) -> int:
    ...

# MTMD_API size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_get_n_tokens", [mtmd_input_chunk_p_ctypes], c_size_t)
def mtmd_input_chunk_get_n_tokens(chunk: mtmd_input_chunk_p, /) -> int:
    ...

# MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk);
@ctypes_function("mtmd_input_chunk_get_type", [mtmd_input_chunk_p_ctypes], c_int)
def mtmd_input_chunk_get_type(chunk: mtmd_input_chunk_p, /) -> int:
    ...

# MTMD_API const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output);
@ctypes_function(
    "mtmd_input_chunk_get_tokens_text",
    [mtmd_input_chunk_p_ctypes, POINTER(c_size_t)],
    POINTER(llama_cpp.llama_token)
)
def mtmd_input_chunk_get_tokens_text(
    chunk: mtmd_input_chunk_p, n_tokens_output: "_Pointer[c_size_t]", /
) -> Optional["_Pointer[llama_cpp.llama_token]"]:
    ...

################################################
# mtmd-helper.h functions
################################################

# MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);
@ctypes_function(
    "mtmd_helper_bitmap_init_from_buf",
    [mtmd_context_p_ctypes, POINTER(c_uint8), c_size_t],
    mtmd_bitmap_p_ctypes
)
def mtmd_helper_bitmap_init_from_buf(
    ctx: mtmd_context_p,
    buf: CtypesArray[c_uint8],
    length: Union[c_size_t, int],
    /,
) -> Optional[mtmd_bitmap_p]:
    ...

# MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);
@ctypes_function("mtmd_helper_get_n_tokens", [mtmd_input_chunks_p_ctypes], c_size_t)
def mtmd_helper_get_n_tokens(chunks: mtmd_input_chunks_p, /) -> int:
    ...

# MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
#                                                struct llama_context * lctx,
#                                                const mtmd_input_chunk * chunk,
#                                                llama_pos n_past,
#                                                llama_seq_id seq_id,
#                                                int32_t n_batch,
#                                                bool logits_last,
#                                                llama_pos * new_n_past);
@ctypes_function(
    "mtmd_helper_eval_chunk_single",
    [
        mtmd_context_p_ctypes,
        llama_cpp.llama_context_p_ctypes,
        mtmd_input_chunk_p_ctypes,
        llama_cpp.llama_pos,
        llama_cpp.llama_seq_id,
        c_int,
        c_bool,
        POINTER(llama_cpp.llama_pos),
    ],
    c_int,
)
def mtmd_helper_eval_chunk_single(
    ctx: mtmd_context_p,
    lctx: llama_cpp.llama_context_p,
    chunk: mtmd_input_chunk_p,
    n_past: llama_cpp.llama_pos,
    seq_id: llama_cpp.llama_seq_id,
    n_batch: Union[c_int, int],
    logits_last: Union[c_bool, bool],
    new_n_past: "_Pointer[llama_cpp.llama_pos]",
    /,
) -> int:
    ...
