// Explicit layers run below NVIDIA's implicit Smooth Motion layer, so this
// counter includes its generated frames. It does not alter any Vulkan call.
// https://forums.developer.nvidia.com/t/580-release-feedback-discussion/341205/329
// Loader dispatch and negotiation:
// https://vulkan.lunarg.com/doc/view/latest/linux/LoaderLayerInterface.html
#include <vulkan/vk_layer.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>

namespace {
struct Instance {
  VkInstance handle;
  PFN_vkGetInstanceProcAddr gipa;
  PFN_GetPhysicalDeviceProcAddr physical;
};
struct Device {
  PFN_vkGetDeviceProcAddr gdpa;
  PFN_vkQueuePresentKHR present;
};
std::mutex registry_mutex;
std::unordered_map<void *, Instance> instances;
std::unordered_map<void *, Device> devices;
std::atomic<uint64_t> presentations{};
std::atomic<uint32_t> attached_devices{};

template <typename T> void *key(T object) {
  return *reinterpret_cast<void **>(object);
}
Instance instance_for(void *dispatch) {
  std::lock_guard lock(registry_mutex);
  return instances.at(dispatch);
}
Device device_for(void *dispatch) {
  std::lock_guard lock(registry_mutex);
  return devices.at(dispatch);
}
} // namespace

extern "C" {
VK_LAYER_EXPORT uint64_t nixePresentationCount() {
  return presentations.load(std::memory_order_relaxed);
}
VK_LAYER_EXPORT uint32_t nixePresentationCounterAttached() {
  return attached_devices.load(std::memory_order_relaxed);
}

PFN_vkVoidFunction VKAPI_CALL CounterGIPA(VkInstance, const char *);
PFN_vkVoidFunction VKAPI_CALL CounterGDPA(VkDevice, const char *);

VkResult VKAPI_CALL CounterCreateInstance(
    const VkInstanceCreateInfo *info, const VkAllocationCallbacks *allocator,
    VkInstance *output) {
  auto *chain =
      reinterpret_cast<const VkLayerInstanceCreateInfo *>(info->pNext);
  while (chain &&
         (chain->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
          chain->function != VK_LAYER_LINK_INFO)) {
    chain = reinterpret_cast<const VkLayerInstanceCreateInfo *>(chain->pNext);
  }
  if (!chain)
    return VK_ERROR_INITIALIZATION_FAILED;
  const auto next = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  const auto physical = chain->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;
  const_cast<VkLayerInstanceCreateInfo *>(chain)->u.pLayerInfo =
      chain->u.pLayerInfo->pNext;
  const auto create =
      reinterpret_cast<PFN_vkCreateInstance>(next(nullptr, "vkCreateInstance"));
  const auto result = create(info, allocator, output);
  if (result == VK_SUCCESS) {
    try {
      std::lock_guard lock(registry_mutex);
      instances.emplace(key(*output), Instance{*output, next, physical});
    } catch (const std::bad_alloc &) {
      reinterpret_cast<PFN_vkDestroyInstance>(
          next(*output, "vkDestroyInstance"))(*output, allocator);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
  }
  return result;
}
void VKAPI_CALL CounterDestroyInstance(VkInstance instance,
                                       const VkAllocationCallbacks *allocator) {
  const auto dispatch = instance_for(key(instance));
  const auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(
      dispatch.gipa(instance, "vkDestroyInstance"));
  {
    std::lock_guard lock(registry_mutex);
    instances.erase(key(instance));
  }
  destroy(instance, allocator);
}
VkResult VKAPI_CALL CounterCreateDevice(VkPhysicalDevice physical,
                                        const VkDeviceCreateInfo *info,
                                        const VkAllocationCallbacks *allocator,
                                        VkDevice *output) {
  const auto instance = instance_for(key(physical));
  auto *chain = reinterpret_cast<const VkLayerDeviceCreateInfo *>(info->pNext);
  while (chain &&
         (chain->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
          chain->function != VK_LAYER_LINK_INFO)) {
    chain = reinterpret_cast<const VkLayerDeviceCreateInfo *>(chain->pNext);
  }
  if (!chain)
    return VK_ERROR_INITIALIZATION_FAILED;
  const auto gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  const auto gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  const_cast<VkLayerDeviceCreateInfo *>(chain)->u.pLayerInfo =
      chain->u.pLayerInfo->pNext;
  const auto create = reinterpret_cast<PFN_vkCreateDevice>(
      gipa(instance.handle, "vkCreateDevice"));
  const auto result = create(physical, info, allocator, output);
  if (result == VK_SUCCESS) {
    try {
      std::lock_guard lock(registry_mutex);
      devices.emplace(key(*output),
                      Device{gdpa, reinterpret_cast<PFN_vkQueuePresentKHR>(
                                       gdpa(*output, "vkQueuePresentKHR"))});
      attached_devices.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::bad_alloc &) {
      reinterpret_cast<PFN_vkDestroyDevice>(gdpa(*output, "vkDestroyDevice"))(
          *output, allocator);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
  }
  return result;
}
void VKAPI_CALL CounterDestroyDevice(VkDevice device,
                                     const VkAllocationCallbacks *allocator) {
  const auto dispatch = device_for(key(device));
  const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(
      dispatch.gdpa(device, "vkDestroyDevice"));
  {
    std::lock_guard lock(registry_mutex);
    devices.erase(key(device));
    attached_devices.fetch_sub(1, std::memory_order_relaxed);
  }
  destroy(device, allocator);
}
VkResult VKAPI_CALL CounterPresent(VkQueue queue,
                                   const VkPresentInfoKHR *info) {
  const auto dispatch = device_for(key(queue));
  const auto result = dispatch.present(queue, info);
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    return result;
  uint64_t count = 0;
  for (uint32_t i = 0; i < info->swapchainCount; ++i) {
    const auto status = info->pResults ? info->pResults[i] : result;
    if (status == VK_SUCCESS || status == VK_SUBOPTIMAL_KHR)
      ++count;
  }
  presentations.fetch_add(count, std::memory_order_relaxed);
  return result;
}
PFN_vkVoidFunction VKAPI_CALL CounterGDPA(VkDevice device, const char *name) {
  if (!std::strcmp(name, "vkGetDeviceProcAddr"))
    return reinterpret_cast<PFN_vkVoidFunction>(CounterGDPA);
  if (!std::strcmp(name, "vkDestroyDevice"))
    return reinterpret_cast<PFN_vkVoidFunction>(CounterDestroyDevice);
  const auto dispatch = device_for(key(device));
  if (!std::strcmp(name, "vkQueuePresentKHR")) {
    return dispatch.present
               ? reinterpret_cast<PFN_vkVoidFunction>(CounterPresent)
               : nullptr;
  }
  return dispatch.gdpa(device, name);
}
PFN_vkVoidFunction VKAPI_CALL CounterGIPA(VkInstance instance,
                                          const char *name) {
#define ENTRY(api, function)                                                   \
  if (!std::strcmp(name, api))                                                 \
  return reinterpret_cast<PFN_vkVoidFunction>(function)
  ENTRY("vkGetInstanceProcAddr", CounterGIPA);
  ENTRY("vkGetDeviceProcAddr", CounterGDPA);
  ENTRY("vkCreateInstance", CounterCreateInstance);
  ENTRY("vkDestroyInstance", CounterDestroyInstance);
  ENTRY("vkCreateDevice", CounterCreateDevice);
  ENTRY("vkDestroyDevice", CounterDestroyDevice);
#undef ENTRY
  if (!instance)
    return nullptr;
  const auto dispatch = instance_for(key(instance));
  const auto next = dispatch.gipa(instance, name);
  if (next && !std::strcmp(name, "vkQueuePresentKHR")) {
    return reinterpret_cast<PFN_vkVoidFunction>(CounterPresent);
  }
  return next;
}
PFN_vkVoidFunction VKAPI_CALL CounterPhysical(VkInstance instance,
                                              const char *name) {
  const auto dispatch = instance_for(key(instance));
  return dispatch.physical ? dispatch.physical(instance, name)
                           : dispatch.gipa(instance, name);
}
VK_LAYER_EXPORT VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *version) {
  if (version->loaderLayerInterfaceVersion < 2)
    return VK_ERROR_INITIALIZATION_FAILED;
  version->loaderLayerInterfaceVersion = 2;
  version->pfnGetInstanceProcAddr = CounterGIPA;
  version->pfnGetDeviceProcAddr = CounterGDPA;
  version->pfnGetPhysicalDeviceProcAddr = CounterPhysical;
  return VK_SUCCESS;
}
} // extern "C"
